"""
Stacking Ensemble - Classification COVID/Normal/Lung_Opacity/Viral_Pneumonia
=============================================================================

Combine par stacking (meta-apprentissage) :
  - 3 modeles Keras (vgg16_finetuned.keras, densenet.keras, cnn_256.keras)
    qui predisent directement a partir des images de
    COVID-19_Radiography_Dataset_split/{validation,test}/
  - 1 modele SVM (scikit-learn) qui predit a partir des features tabulaires
    de enhanced_features/{validation,test}_features.csv, standardisees
    (StandardScaler)

Principe : au lieu d'un vote majoritaire, une regression logistique (le
"meta-modele") apprend a ponderer les probabilites/scores de sortie de
chaque modele de base pour chaque classe.

  - Meta-features d'un echantillon = concatenation des probabilites (ou
    scores) des 4 modeles pour les 4 classes -> vecteur de 16 valeurs.
  - Le meta-modele est ENTRAINE sur les predictions des modeles de base sur
    le jeu de VALIDATION (jamais vu pendant l'entrainement des modeles de
    base), avec les vrais labels de validation.
  - Il est ensuite EVALUE sur les predictions des modeles de base sur le
    jeu de TEST, jamais vu ni par les modeles de base ni par le meta-modele.

Produit les statistiques de performance (accuracy, f1, rapport de
classification, matrice de confusion) pour chaque modele individuel et pour
l'ensemble par stacking.

A executer en local, la ou se trouvent les modeles et les donnees.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


@keras.saving.register_keras_serializable()
class SparseF1Score(tf.keras.metrics.F1Score):
    """F1Score de Keras adaptee aux labels sparses (entiers) plutot que one-hot."""
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

SVM_MODEL_PATH = "svm_weighted_enhanced_features.joblib"
SVM_SCALER_PATH = "scaler_svm_enhanced_features.joblib"
SVM_FEATURE_COLUMNS_PATH = "svm_feature_columns.json"

# Chemin de sauvegarde du meta-modele entraine (regression logistique)
META_MODEL_PATH = "stacking_meta_model.joblib"


def load_svm_feature_columns(path=SVM_FEATURE_COLUMNS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable - sauvegarde X_train.columns.tolist() depuis "
            "le notebook SVM."
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


def reorder_columns(matrix, source_labels, canonical_labels):
    """
    Reordonne les colonnes de `matrix` (N, len(source_labels)) pour qu'elles
    suivent l'ordre de `canonical_labels`, en faisant correspondre les
    labels via leur forme normalisee (insensible a la casse/espaces).
    """
    idx = []
    for c in canonical_labels:
        norm_c = normalize_label(c)
        matches = [i for i, s in enumerate(source_labels) if normalize_label(s) == norm_c]
        if not matches:
            raise ValueError(f"Classe '{c}' introuvable parmi {list(source_labels)}")
        idx.append(matches[0])
    return matrix[:, idx]


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
    Charge enhanced_features/{split}_features.csv et reordonne ses lignes
    pour correspondre exactement a l'ordre de file_paths (meme ordre que les
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
            "Modele ou scaler SVM introuvable - voir load_svm_feature_columns "
            "et les instructions precedentes pour les sauvegarder."
        )
    return joblib.load(model_path), joblib.load(scaler_path)


# ---------------------------------------------------------------------------
# Preprocessing specifique par modele Keras (idem version hard voting)
# ---------------------------------------------------------------------------

def _get_extra_preprocess(model_name):
    name = model_name.lower()
    if "vgg16" in name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    return lambda x: x  # identite par defaut


def adapt_dataset_for_model(ds, model, model_name, fallback_shape=(*IMG_SIZE, 1)):
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


# ---------------------------------------------------------------------------
# Sorties "brutes" des modeles de base (probas/scores par classe, alignees
# sur l'ordre canonique de classes) -> ce sont les meta-features du stacking
# ---------------------------------------------------------------------------

def keras_probs(models, ds, class_names, canonical_classes):
    """{nom_modele: array (N, n_classes) de probas, colonnes = canonical_classes}."""
    probs_by_model = {}
    for name, model in models.items():
        adapted_ds = adapt_dataset_for_model(ds, model, name)
        probs = model.predict(adapted_ds, verbose=0)
        probs_by_model[name] = reorder_columns(probs, class_names, canonical_classes)
    return probs_by_model


def svm_scores(svm_model, scaler, features_df, feature_columns, canonical_classes):
    """
    Scores continus par classe pour le SVM, colonnes = canonical_classes.
    Utilise predict_proba si le SVM a ete entraine avec probability=True,
    sinon decision_function (disponible sans reentrainement, fonctionne
    aussi bien comme meta-feature pour la regression logistique).
    """
    missing = [c for c in feature_columns if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} colonne(s) de features attendues par le SVM sont absentes "
            f"du CSV de features charge : {missing}\n"
            "Regenere les CSV validation/test avec la meme version de "
            "generate_features_csv_v6_cma.ipynb que train_features.csv."
        )
    X_scaled = scaler.transform(features_df[feature_columns])

    if hasattr(svm_model, "predict_proba") and getattr(svm_model, "probability", False):
        scores = svm_model.predict_proba(X_scaled)
    else:
        # decision_function : fonctionne sans probability=True. Pour un SVC
        # multiclasse avec decision_function_shape='ovr' (par defaut), la
        # forme est deja (N, n_classes).
        scores = svm_model.decision_function(X_scaled)
        if scores.ndim == 1:
            raise ValueError(
                "decision_function renvoie un vecteur 1D : verifie que le SVM a bien "
                "ete entraine en multiclasse avec decision_function_shape='ovr' (defaut)."
            )

    return reorder_columns(scores, svm_model.classes_, canonical_classes)


# ---------------------------------------------------------------------------
# Construction des meta-features et stacking
# ---------------------------------------------------------------------------

def build_meta_features(canonical_classes, kprobs, sscores):
    """
    Concatene les sorties de tous les modeles de base en une seule matrice
    (N, n_modeles * n_classes), dans un ordre de colonnes stable.
    """
    model_names = sorted(kprobs.keys()) + ["svm"]
    parts = [kprobs[name] for name in sorted(kprobs.keys())] + [sscores]
    X_meta = np.concatenate(parts, axis=1)
    feature_names = [f"{name}_{c}" for name in model_names for c in canonical_classes]
    return X_meta, feature_names


def train_meta_model(X_meta_val, y_val):
    """Regression logistique multiclasse entrainee sur les meta-features de validation."""
    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta_model.fit(X_meta_val, y_val)
    return meta_model


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

    out_path = Path(out_dir) / f"confusion_matrices_{split}_stacking.png"
    fig.savefig(out_path, dpi=150)
    print(f"Matrices de confusion sauvegardees dans {out_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline par split : calcule les meta-features + labels vrais
# ---------------------------------------------------------------------------

def compute_split_data(split, keras_models, svm_model, svm_scaler, svm_feature_columns, canonical_classes):
    print(f"\n########## {split.upper()} - calcul des predictions de base ##########\n")
    ds, file_paths, class_names, y_true_idx = load_image_dataset(split)
    y_true = [class_names[i] for i in y_true_idx]

    features_df = load_features_aligned(split, file_paths)

    kprobs = keras_probs(keras_models, ds, class_names, canonical_classes)
    sscores = svm_scores(svm_model, svm_scaler, features_df, svm_feature_columns, canonical_classes)

    X_meta, feature_names = build_meta_features(canonical_classes, kprobs, sscores)

    # Predictions "argmax" de chaque modele individuel, pour comparaison
    y_preds_individual = {
        name: [canonical_classes[i] for i in np.argmax(probs, axis=1)]
        for name, probs in kprobs.items()
    }
    y_preds_individual["svm"] = [canonical_classes[i] for i in np.argmax(sscores, axis=1)]

    return {
        "y_true": y_true,
        "X_meta": X_meta,
        "feature_names": feature_names,
        "y_preds_individual": y_preds_individual,
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    keras_models = load_keras_models(KERAS_MODEL_PATHS)
    svm_model, svm_scaler = load_svm(SVM_MODEL_PATH, SVM_SCALER_PATH)
    svm_feature_columns = load_svm_feature_columns()

    # Ordre canonique des classes, fixe une fois pour toutes (celui du jeu
    # de validation, suppose identique a celui du jeu de test).
    _, _, canonical_classes, _ = load_image_dataset("validation")

    val_data = compute_split_data(
        "validation", keras_models, svm_model, svm_scaler, svm_feature_columns, canonical_classes)
    test_data = compute_split_data(
        "test", keras_models, svm_model, svm_scaler, svm_feature_columns, canonical_classes)

    # --- Entrainement du meta-modele sur les meta-features de VALIDATION ---
    meta_model = train_meta_model(val_data["X_meta"], val_data["y_true"])
    joblib.dump(meta_model, META_MODEL_PATH)
    print(f"\nMeta-modele (regression logistique) sauvegarde dans {META_MODEL_PATH}")

    # --- Evaluation sur VALIDATION (optimiste : deja vu par le meta-modele) ---
    print("\n" + "=" * 80)
    print("ATTENTION : l'evaluation sur validation ci-dessous est optimiste, "
          "le meta-modele a ete entraine sur ces memes donnees. "
          "Seule l'evaluation sur test ci-apres est representative.")
    print("=" * 80)
    y_pred_stacking_val = meta_model.predict(val_data["X_meta"])
    val_preds = dict(val_data["y_preds_individual"])
    val_preds["stacking"] = list(y_pred_stacking_val)

    val_summary = {
        name: evaluate(val_data["y_true"], y_pred, val_data["class_names"], f"validation - {name}")
        for name, y_pred in val_preds.items()
    }
    plot_confusion_grid(val_data["y_true"], val_preds, val_data["class_names"], "validation")

    # --- Evaluation finale sur TEST ---
    y_pred_stacking_test = meta_model.predict(test_data["X_meta"])
    test_preds = dict(test_data["y_preds_individual"])
    test_preds["stacking"] = list(y_pred_stacking_test)

    test_summary = {
        name: evaluate(test_data["y_true"], y_pred, test_data["class_names"], f"test - {name}")
        for name, y_pred in test_preds.items()
    }
    plot_confusion_grid(test_data["y_true"], test_preds, test_data["class_names"], "test")

    print("\n--- Recapitulatif TEST (evaluation representative) ---")
    for name, m in test_summary.items():
        print(f"{name:15s} accuracy={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")
