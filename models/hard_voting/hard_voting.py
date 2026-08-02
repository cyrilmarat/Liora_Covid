"""
Hard Voting Ensemble - Classification COVID/Normal/Lung_Opacity/Viral_Pneumonia
=================================================================================

Combine par vote majoritaire (hard voting) :
  - 3 modeles Keras (vgg16_finetuned.keras, densenet.keras, cnn_256.keras)
    qui predisent directement a partir des images de
    COVID-19_Radiography_Dataset_split/{validation,test}/
  - 1 modele SVM (scikit-learn) qui predit a partir des features tabulaires
    de features/{validation,test}_features.csv, standardisees (StandardScaler)

Produit egalement les statistiques de performance (accuracy, f1, rapport de
classification, matrice de confusion) pour chaque modele individuel et pour
l'ensemble, sur les jeux de validation et de test.

A executer en local, la ou se trouvent les modeles et les donnees.
"""

from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import keras

@keras.saving.register_keras_serializable()
class SparseF1Score(tf.keras.metrics.F1Score):
    """F1Score de Keras adaptée aux labels sparses (entiers) plutôt que one-hot."""
    def __init__(self, num_classes=4, **kwargs):
        super().__init__(**kwargs)
        self.num_classes_ = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=self.num_classes_)
        return super().update_state(y_true, y_pred, sample_weight)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DATA_DIR = Path("../../../COVID-19_Radiography_Dataset_split")
FEATURES_DIR = Path("../../../enhanced_features")

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

KERAS_MODEL_PATHS = {
    "vgg16": "vgg16_finetuned.keras",
    "densenet": "densenet.keras",
    "cnn_256": "cnn_256.keras",
}

# Le notebook svm_weighted_enhanced_features.ipynb ne sauvegarde actuellement
# que y_pred (.npy). Il faut ajouter a la fin de ce notebook :
#   joblib.dump(grid_clf.best_estimator_, "svm_weighted_enhanced_features.joblib")
#   joblib.dump(scaler, "scaler_svm_enhanced_features.joblib")
SVM_MODEL_PATH = "svm_weighted_enhanced_features.joblib"
SVM_SCALER_PATH = "scaler_svm_enhanced_features.joblib"

# Ordre exact des colonnes de features utilise a l'entrainement du SVM,
# sauvegarde depuis le notebook via :
#   json.dump(X_train.columns.tolist(), open("svm_feature_columns.json", "w"))
SVM_FEATURE_COLUMNS_PATH = "svm_feature_columns.json"


def load_svm_feature_columns(path=SVM_FEATURE_COLUMNS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable - sauvegarde X_train.columns.tolist() depuis "
            "le notebook SVM (cf. commentaire ci-dessus)."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Normalisation des labels (le SVM utilise les valeurs de la colonne
# 'classe', les datasets Keras utilisent les noms de sous-dossiers : on les
# aligne sur une forme canonique commune)
# ---------------------------------------------------------------------------

def normalize_label(label):
    return str(label).strip().lower().replace(" ", "_")


def build_label_lookup(class_names):
    """class_names : labels canoniques de reference (ex. test_ds.class_names)."""
    return {normalize_label(c): c for c in class_names}


# ---------------------------------------------------------------------------
# Chargement des images (identique au pipeline d'entrainement)
# ---------------------------------------------------------------------------

def load_image_dataset(split):
    """split : 'validation' ou 'test'."""
    seed_kwarg = {"seed": 42} if split == "test" else {}
    ds = image_dataset_from_directory(
        directory=str(DATA_DIR / split),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels="inferred",
        shuffle=False,
        color_mode="grayscale",
        **seed_kwarg,
    )
    file_paths = ds.file_paths  # ordre identique a l'iteration (shuffle=False)
    class_names = ds.class_names
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
    return ds, file_paths, class_names, y_true


# ---------------------------------------------------------------------------
# Chargement et alignement des features (SVM)
# ---------------------------------------------------------------------------

def load_features_aligned(split, file_paths):
    """
    Charge features/{split}_features.csv et reordonne ses lignes pour
    correspondre exactement a l'ordre de file_paths (meme ordre que les
    predictions Keras), en matchant sur le nom de fichier.
    """
    csv_path = FEATURES_DIR / f"{split}_features.csv"
    df = pd.read_csv(csv_path, sep=",")

    filenames = [Path(p).name for p in file_paths]
    df = df.set_index("filename").loc[filenames].reset_index()

    missing = df["filename"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} image(s) de {split} sans ligne de features correspondante "
            f"dans {csv_path.name} - verifie la correspondance des noms de fichiers."
        )
    return df


# ---------------------------------------------------------------------------
# Chargement des modeles
# ---------------------------------------------------------------------------

def load_keras_models(model_paths):
    models = {}
    for name, path in model_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modele introuvable : {path}")
        print(f"Chargement de {path.name} ...")
        models[name] = keras.models.load_model(path)
    return models


def load_svm(model_path, scaler_path):
    model_path, scaler_path = Path(model_path), Path(scaler_path)
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Modele ou scaler SVM introuvable - voir le commentaire en tete "
            "de ce script pour les sauvegarder depuis le notebook."
        )
    return joblib.load(model_path), joblib.load(scaler_path)


# ---------------------------------------------------------------------------
# Votes de chaque modele
# ---------------------------------------------------------------------------

# Preprocessing specifique par modele, applique APRES resize/adaptation de
# canaux et AVANT predict. VGG16 attend le preprocess_input Caffe-style
# (BGR + soustraction de la moyenne ImageNet) - ce n'est pas integre au
# .keras sauvegarde, cf. plot_model_vgg16.ipynb. Les autres modeles
# (densenet, cnn_256) n'en ont pas besoin ici (leurs scores matchent deja
# leurs notebooks d'evaluation individuels).
def _get_extra_preprocess(model_name):
    name = model_name.lower()
    if "vgg16" in name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    return lambda x: x  # identite par defaut


def adapt_dataset_for_model(ds, model, model_name, fallback_shape=(*IMG_SIZE, 1)):
    """
    Redimensionne les images et adapte le nombre de canaux (grayscale <-> RGB)
    a la forme d'entree attendue par `model` (model.input_shape), puisque les
    modeles de l'ensemble n'ont pas tous ete entraines avec la meme taille /
    le meme nombre de canaux (ex. vgg16_finetuned en 224x224x3, d'autres en
    299x299x1). Applique ensuite le preprocessing specifique du modele
    (cf. _get_extra_preprocess).
    """
    try:
        _, target_h, target_w, target_c = model.input_shape
    except (AttributeError, TypeError, ValueError):
        target_h, target_w, target_c = fallback_shape

    extra_preprocess = _get_extra_preprocess(model_name)

    def _adapt(img, label):
        img = tf.image.resize(img, (target_h, target_w))
        current_c = img.shape[-1]
        if target_c == 3 and current_c == 1:
            img = tf.image.grayscale_to_rgb(img)
        elif target_c == 1 and current_c == 3:
            img = tf.image.rgb_to_grayscale(img)
        img = extra_preprocess(img)
        return img, label

    return ds.map(_adapt)


def keras_votes(models, ds, class_names, label_lookup):
    """{nom_modele: [(classe_canonique, proba), ...]} dans l'ordre du dataset."""
    votes = {}
    for name, model in models.items():
        adapted_ds = adapt_dataset_for_model(ds, model, name)
        probs = model.predict(adapted_ds, verbose=0)
        votes[name] = [
            (label_lookup[normalize_label(class_names[int(np.argmax(p))])], float(np.max(p)))
            for p in probs
        ]
    return votes


def svm_votes(svm_model, scaler, features_df, label_lookup, feature_columns):
    missing = [c for c in feature_columns if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} colonne(s) de features attendues par le SVM sont absentes "
            f"du CSV de features charge : {missing}\n"
            "Le CSV de features (validation/test) n'a pas ete genere avec la meme "
            "version de generate_features_csv_v6_cma.ipynb que train_features.csv "
            "(utilise pour entrainer le SVM). Regenere les CSV validation/test avec "
            "la version actuelle du notebook de features pour avoir les memes colonnes."
        )
    X_scaled = scaler.transform(features_df[feature_columns])

    preds = svm_model.predict(X_scaled)
    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(X_scaled)
        classes = list(svm_model.classes_)
        confidences = [probs[i, classes.index(pred)] for i, pred in enumerate(preds)]
    else:
        confidences = [1.0] * len(preds)

    return [(label_lookup[normalize_label(p)], float(c)) for p, c in zip(preds, confidences)]


# ---------------------------------------------------------------------------
# Hard voting
# ---------------------------------------------------------------------------

def hard_voting(keras_model_votes, svm_model_votes, n_images):
    results = []
    for i in range(n_images):
        per_model = {name: preds[i] for name, preds in keras_model_votes.items()}
        per_model["svm"] = svm_model_votes[i]

        votes = [label for label, _ in per_model.values()]
        vote_counts = Counter(votes)
        max_votes = max(vote_counts.values())
        tied = [c for c, v in vote_counts.items() if v == max_votes]

        if len(tied) == 1:
            final_class = tied[0]
        else:
            avg_conf = {c: np.mean([conf for lbl, conf in per_model.values() if lbl == c]) for c in tied}
            final_class = max(avg_conf, key=avg_conf.get)

        results.append({"prediction": final_class, "votes": dict(vote_counts), "per_model": per_model})
    return results


# ---------------------------------------------------------------------------
# Statistiques de performance
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, class_names, titre):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"===== {titre} =====")
    print(f"Accuracy : {acc:.4f}   F1-macro : {f1_macro:.4f}\n")
    print(classification_report(y_true, y_pred, labels=class_names, zero_division=0))

    return {"accuracy": acc, "f1_macro": f1_macro}


def plot_confusion_grid(y_true, preds_by_model, class_names, split, out_dir="."):
    """
    Une figure unique par split, avec une matrice de confusion (normalisee)
    par modele + l'ensemble, cote a cote. Sauvegardee en PNG plutot
    qu'affichee en boucle (plus pratique en execution script).

    preds_by_model : dict {nom_modele: y_pred}, ex.
        {"vgg16": [...], "densenet": [...], "cnn_256": [...], "svm": [...], "hard_voting": [...]}
    """
    n = len(preds_by_model)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, preds_by_model.items()):
        cm = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
            ax=ax, xticks_rotation=45, colorbar=False
        )
        ax.set_title(name)

    fig.suptitle(f"Matrices de confusion - {split}")
    fig.tight_layout()

    out_path = Path(out_dir) / f"confusion_matrices_{split}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Matrices de confusion sauvegardees dans {out_path}")
    plt.show()
    plt.close(fig)


def evaluate_split(split, keras_models, svm_model, svm_scaler, svm_feature_columns):
    print(f"\n########## {split.upper()} ##########\n")
    ds, file_paths, class_names, y_true_idx = load_image_dataset(split)
    label_lookup = build_label_lookup(class_names)
    y_true = [class_names[i] for i in y_true_idx]

    features_df = load_features_aligned(split, file_paths)

    kvotes = keras_votes(keras_models, ds, class_names, label_lookup)
    svotes = svm_votes(svm_model, svm_scaler, features_df, label_lookup, svm_feature_columns)

    results = hard_voting(kvotes, svotes, n_images=len(file_paths))
    y_pred_ensemble = [r["prediction"] for r in results]

    y_preds = {name: [label for label, _ in preds] for name, preds in kvotes.items()}
    y_preds["svm"] = [label for label, _ in svotes]
    y_preds["hard_voting"] = y_pred_ensemble

    summary = {
        name: evaluate(y_true, y_pred, class_names, f"{split} - {name}")
        for name, y_pred in y_preds.items()
    }

    plot_confusion_grid(y_true, y_preds, class_names, split)

    print(f"\n--- Recapitulatif {split} ---")
    for name, m in summary.items():
        print(f"{name:15s} accuracy={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")

    return results, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    keras_models = load_keras_models(KERAS_MODEL_PATHS)
    svm_model, svm_scaler = load_svm(SVM_MODEL_PATH, SVM_SCALER_PATH)
    svm_feature_columns = load_svm_feature_columns()

    val_results, val_summary = evaluate_split(
        "validation", keras_models, svm_model, svm_scaler, svm_feature_columns)
    test_results, test_summary = evaluate_split(
        "test", keras_models, svm_model, svm_scaler, svm_feature_columns)
