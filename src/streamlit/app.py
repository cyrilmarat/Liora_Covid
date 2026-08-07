import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
import joblib
import base64
import sys
import json
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins de base (indépendants du répertoire de travail courant)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent          # .../src/streamlit
PROJECT_ROOT = BASE_DIR.parent.parent                # racine du dépôt

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from frozen_results_data import CLASS_NAMES as FROZEN_CLASS_NAMES, LOGREG_RESULTS, XGBOOST_RESULTS

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
model_CNN1024_path = str(PROJECT_ROOT / "models" / "cnn1024" / "cnn_1024.keras")
model_CNN512_path = str(PROJECT_ROOT / "models" / "cnn512" / "cnn_512.keras")
model_CNN256_path = str(PROJECT_ROOT / "models" / "cnn256" / "cnn_256.keras")
model_CNN256_tuned_path = str(PROJECT_ROOT / "models" / "cnn256_tuned" / "cnn_256_tuned.keras")
model_CNN128_path = str(PROJECT_ROOT / "models" / "cnn128" / "cnn_128.keras")
model_CNN64_path = str(PROJECT_ROOT / "models" / "cnn64" / "cnn_64.keras")
model_CNN32_path = str(PROJECT_ROOT / "models" / "cnn32" / "cnn_32.keras")
model_VGG16_head_path = str(PROJECT_ROOT / "models" / "vgg16" / "vgg16_head.keras")
courbe_VGG16_path = str(PROJECT_ROOT / "models" / "vgg16" / "vgg16_f1_combined.png")
model_VGG16_finetuned_path = str(PROJECT_ROOT / "models" / "vgg16" / "vgg16_finetuned.keras")
model_InceptionV3_head_path = str(PROJECT_ROOT / "models" / "inceptionv3" / "inceptionv3_head.keras")
model_InceptionV3_finetuned_path = str(PROJECT_ROOT / "models" / "inceptionv3" / "inceptionv3_finetuned.keras")
courbe_Inceptionv3_path = str(PROJECT_ROOT / "models" / "inceptionv3" / "inceptionv3_f1_combined.png")
model_DenseNet_path = str(PROJECT_ROOT / "models" / "DenseNet" / "densenet.keras")
courbe_DenseNet_path = str(PROJECT_ROOT / "models" / "DenseNet" / "densenet121_f1_combined.png")
val_dir = str(PROJECT_ROOT / "COVID-19_Radiography_Dataset_split" / "validation")
test_dir = str(PROJECT_ROOT / "COVID-19_Radiography_Dataset_split" / "test")
model_SVM_Weighted_path = str(PROJECT_ROOT / "models" / "svm" / "svm_weighted.joblib")
model_SVM_path = str(PROJECT_ROOT / "models" / "svm" / "svm.joblib")
scaler_SVM_path = str(PROJECT_ROOT / "models" / "svm" / "scaler_svm.joblib")
scaler_SVM_Weighted_path = str(PROJECT_ROOT / "models" / "svm" / "scaler_svm_weighted.joblib")
csv_test = str(PROJECT_ROOT / "csv" / "test_features.csv")
csv_validation = str(PROJECT_ROOT / "csv" / "validation_features.csv")
model_random_forest_Weighted_path = str(PROJECT_ROOT / "models" / "random_forest" / "random_forest.joblib")

# Répertoire de stockage en dur des résultats calculés (val/test) par modèle.
# Remplace le cache mémoire (perdu au redémarrage de l'app) par des fichiers
# JSON persistants sur disque, un par modèle/variante.
RESULTS_DIR = PROJECT_ROOT / "results_cache"
RESULTS_DIR.mkdir(exist_ok=True)

courbe_CNN1024_path = str(PROJECT_ROOT / "models" / "cnn1024" / "cnn_1024.png")
courbe_CNN512_path = str(PROJECT_ROOT / "models" / "cnn512" / "cnn_512.png")
courbe__CNN256_path = str(PROJECT_ROOT / "models" / "cnn256" / "cnn_256.png")
courbe_CNN256_tuned_path = str(PROJECT_ROOT / "models" / "cnn256_tuned" / "cnn_256_tuned.png")
courbe__CNN128_path = str(PROJECT_ROOT / "models" / "cnn128" / "cnn_128.png")
courbe_CNN64_path = str(PROJECT_ROOT / "models" / "cnn64" / "cnn_64.png")
courbe_CNN32_path = str(PROJECT_ROOT / "models" / "cnn32" / "cnn_32.png")


# --------------------------------------------------------------------------- #
# Definitions globales
# --------------------------------------------------------------------------- #
size_img=299

# --------------------------------------------------------------------------- #
# Métrique custom nécessaire pour désérialiser le modèle 
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable()
class SparseF1Score(tf.keras.metrics.F1Score):
    """F1Score de Keras adaptée aux labels sparses (entiers) plutôt que one-hot."""

    def __init__(self, num_classes=4, **kwargs):
        super().__init__(**kwargs)
        self.num_classes_ = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=self.num_classes_)
        return super().update_state(y_true, y_pred, sample_weight)


# --------------------------------------------------------------------------- #
# Config page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title=" Analyse des radiographies pulmonaires", layout="wide")
st.title("🫁 Analyse des radiographies pulmonaires 🫁 ")
st.caption("Classification COVID / NORMAL / LUNG_OPACITY / VIRAL_PNEUMONIA  | https://lioracovid.streamlit.app")




# --------------------------------------------------------------------------- #
# Utilitaire pour l'animation du pré-processing
# --------------------------------------------------------------------------- #
def image_to_data_uri(image_path: str) -> str:
    """Convertit une image locale en Data URI pour l'afficher dans un composant HTML."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    extension = path.suffix.lower().replace(".", "")
    mime_type = "jpeg" if extension in {"jpg", "jpeg"} else extension

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime_type};base64,{encoded}"


# --------------------------------------------------------------------------- #
# Fonctions mises en cache
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Chargement du modèle CNN1024")
def get_model_CNN1024():
    return load_model(model_CNN1024_path)

@st.cache_resource(show_spinner="Chargement du modèle CNN512")
def get_model_CNN512():
    return load_model(model_CNN512_path)

@st.cache_resource(show_spinner="Chargement du modèle CNN256")
def get_model_CNN256():
    return load_model(model_CNN256_path)

@st.cache_resource(show_spinner="Chargement du modèle CNN256 Tuned")
def get_model_CNN256_tuned():
    return load_model(model_CNN256_tuned_path)

@st.cache_resource(show_spinner="Chargement du modèle CNN128")
def get_model_CNN128():
    return load_model(model_CNN128_path)

@st.cache_resource(show_spinner="Chargement du modèle CNN64")
def get_model_CNN64():
    return load_model(model_CNN64_path)

@st.cache_resource(show_spinner="Chargement du modèle CNN32")
def get_model_CNN32():
    return load_model(model_CNN32_path)

@st.cache_resource(show_spinner="Chargement du modèle VGG16 (head) …")
def get_model_VGG16_head():
    return load_model(model_VGG16_head_path)

@st.cache_resource(show_spinner="Chargement du modèle VGG16 (fine-tuned) …")
def get_model_VGG16_finetuned():
    return load_model(model_VGG16_finetuned_path)

@st.cache_resource(show_spinner="Chargement du modèle InceptionV3 (head) …")
def get_model_InceptionV3_head():
    return load_model(model_InceptionV3_head_path)

@st.cache_resource(show_spinner="Chargement du modèle InceptionV3 (fine-tuned) …")
def get_model_InceptionV3_finetuned():
    return load_model(model_InceptionV3_finetuned_path)

@st.cache_resource(show_spinner="Chargement du modèle DenseNet …")
def get_model_DenseNet():
    return load_model(model_DenseNet_path)

@st.cache_resource(show_spinner="Chargement du modèle SVM …")
def get_model_SVM(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner="Chargement du scaler SVM …")
def get_scaler_SVM(path: str):
    return joblib.load(path)



@st.cache_resource(show_spinner="Chargement du jeu de données…")
def get_dataset(directory: str, img_h: int, img_w: int, batch_size: int, color_mode: str):
    return image_dataset_from_directory(
        directory=directory,
        image_size=(img_h, img_w),
        batch_size=batch_size,
        labels="inferred",
        shuffle=False,
        color_mode=color_mode,
    )


@st.cache_resource(show_spinner="Chargement du jeu de données…")
def get_dataset_dl(directory: str, img_h: int, img_w: int, batch_size: int, color_mode: str, _preprocess_fn=None):
    """Charge un jeu de données pour un modèle de Deep Learning.

    `_preprocess_fn` (préfixé par `_` pour ne pas être hashé par le cache de
    Streamlit) applique le preprocessing spécifique au réseau de transfer
    learning (VGG16, InceptionV3, DenseNet). Laisser à None pour les CNN
    maison, qui gèrent leur normalisation en interne.
    """
    ds = image_dataset_from_directory(
        directory=directory,
        image_size=(img_h, img_w),
        batch_size=batch_size,
        labels="inferred",
        shuffle=False,
        color_mode=color_mode,
    )
    class_names = ds.class_names
    if _preprocess_fn is not None:
        ds = ds.map(lambda x, y: (_preprocess_fn(x), y))
    return ds, class_names


def evaluer(y_true, y_pred, class_names, titre):
    """Affiche accuracy, F1-macro, rappel COVID, classification_report et
    matrice de confusion dans Streamlit."""
    acc = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    f1_macro = report_dict["macro avg"]["f1-score"]

    # Recherche insensible à la casse/accents du libellé de la classe COVID,
    # pour rester robuste quel que soit le nommage exact des dossiers/labels.
    nom_covid = next(
        (nom for nom in class_names if "covid" in nom.lower()), None
    )
    rappel_covid = report_dict[nom_covid]["recall"] if nom_covid is not None else None
    f1_covid = report_dict[nom_covid]["f1-score"] if nom_covid is not None else None

    st.subheader(titre)
    col_acc, col_f1, col_f1_covid, col_covid = st.columns(4)
    col_acc.metric("Accuracy", f"{acc:.4f}")
    col_f1.metric("F1-score macro", f"{f1_macro:.4f}")
    col_f1_covid.metric(
        "F1-score COVID",
        f"{f1_covid:.4f}" if f1_covid is not None else "N/A",
    )
    col_covid.metric(
        "Rappel COVID",
        f"{rappel_covid:.4f}" if rappel_covid is not None else "N/A",
    )

    st.dataframe(pd.DataFrame(report_dict).transpose().round(3))

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    # Format compact pour éviter que la matrice n'occupe toute la largeur.
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    cm_display.plot(ax=ax, colorbar=False, values_format=".2f")

    ax.tick_params(axis="both", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_xlabel("Classe prédite", fontsize=9)
    ax.set_ylabel("Classe réelle", fontsize=9)
    plt.tight_layout()

    # Affichage centré dans une colonne plus étroite.
    col_gauche, col_matrice, col_droite = st.columns([1, 1.35, 1])
    with col_matrice:
        st.pyplot(fig, width="stretch")

    plt.close(fig)

    return acc


def evaluer_depuis_matrice(matrice_confusion, class_names, titre):
    """Reconstitue y_true/y_pred à partir d'une matrice de confusion figée puis
    réutilise `evaluer` pour un rendu identique aux pages en inférence live
    (SVM, CNN), sans dépendre du modèle ni de la version des librairies."""
    y_true, y_pred = [], []
    for i, vrai_label in enumerate(class_names):
        for j, label_predit in enumerate(class_names):
            effectif = matrice_confusion[i][j]
            y_true.extend([vrai_label] * effectif)
            y_pred.extend([label_predit] * effectif)

    return evaluer(pd.Series(y_true), pd.Series(y_pred), class_names, titre)


# --------------------------------------------------------------------------- #
# Stockage en dur des résultats calculés (remplace le cache mémoire)
# --------------------------------------------------------------------------- #
def _cle_fichier_resultats(model_key: str) -> Path:
    """Construit un nom de fichier sûr (sans espaces/accents/caractères spéciaux)
    à partir de la clé du modèle, pour servir de nom de fichier JSON."""
    safe = "".join(c if c.isalnum() else "_" for c in model_key).strip("_").lower()
    return RESULTS_DIR / f"{safe}.json"


def charger_resultats_stockes(model_key: str):
    """Charge les résultats (val/test) précédemment stockés en dur pour ce modèle.

    Renvoie None si aucun résultat n'a encore été calculé/stocké, ou si le
    fichier est illisible."""
    chemin = _cle_fichier_resultats(model_key)
    if not chemin.exists():
        return None
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def sauvegarder_resultats(model_key, class_names, val_y_true, val_y_pred, test_y_true, test_y_pred) -> Path:
    """Stocke en dur (fichier JSON sur disque) les prédictions val/test d'un
    modèle, pour ne plus avoir à refaire l'inférence à chaque affichage de
    page. Écrase le fichier existant s'il y en avait déjà un."""
    donnees = {
        "class_names": list(class_names),
        "calcule_le": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "validation": {
            "y_true": np.asarray(val_y_true).tolist(),
            "y_pred": np.asarray(val_y_pred).tolist(),
        },
        "test": {
            "y_true": np.asarray(test_y_true).tolist(),
            "y_pred": np.asarray(test_y_pred).tolist(),
        },
    }
    chemin = _cle_fichier_resultats(model_key)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(donnees, f)
    return chemin


# --------------------------------------------------------------------------- #
# Stockage en dur du résumé (architecture) d'un modèle Keras
# --------------------------------------------------------------------------- #
def _cle_fichier_summary(model_key: str) -> Path:
    """Construit le chemin du fichier texte stockant le résumé (summary) d'un
    modèle, à partir de la même clé que celle utilisée pour les résultats."""
    safe = "".join(c if c.isalnum() else "_" for c in model_key).strip("_").lower()
    return RESULTS_DIR / f"{safe}_summary.txt"


def charger_summary_stocke(model_key: str):
    """Charge le résumé du modèle précédemment stocké en dur.

    Renvoie None si aucun résumé n'a encore été stocké, ou si le fichier est
    illisible — sans jamais lever d'exception."""
    chemin = _cle_fichier_summary(model_key)
    if not chemin.exists():
        return None
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def sauvegarder_summary(model_key: str, summary_text: str) -> Path:
    """Stocke en dur (fichier texte) le résumé (architecture) d'un modèle
    Keras, pour pouvoir l'afficher même quand le modèle n'est plus
    chargeable. Écrase le fichier existant s'il y en avait déjà un."""
    chemin = _cle_fichier_summary(model_key)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return chemin


def afficher_section_avec_calcul(model_key: str, fonction_calcul, disabled: bool = False, summary_text: str | None = None):
    """Affiche les résultats val/test d'un modèle avec un bouton Calculer/Recalculer.

    Les résultats stockés sur disque (s'ils existent) sont affichés directement,
    sans refaire l'inférence. `fonction_calcul` est un callable sans argument,
    appelé uniquement au clic sur le bouton (ou s'il n'y a encore aucun résultat
    stocké et que l'utilisateur clique) ; il doit renvoyer
    (class_names, val_y_true, val_y_pred, test_y_true, test_y_pred). Le résultat
    est alors stocké en dur et écrase le fichier existant.

    `disabled` grise le bouton (typiquement quand le modèle Keras n'a pas pu
    être chargé) : le recalcul est alors impossible tant que le modèle n'est
    pas de nouveau disponible. `summary_text`, s'il est fourni, est stocké en
    dur au clic sur le bouton, en plus des résultats — c'est ce texte qui est
    ensuite ré-affiché dans « Résumé du modèle » si le modèle devient
    indisponible."""
    resultats = charger_resultats_stockes(model_key)

    col_bouton, col_info = st.columns([1, 3])
    with col_bouton:
        clic = st.button(
            "🔄 Recalculer" if resultats else "▶️ Calculer",
            key=f"calc_btn_{model_key}",
            disabled=disabled,
        )
    with col_info:
        if disabled:
            st.caption("Modèle indisponible — recalcul désactivé.")
        elif resultats:
            st.caption(f"Résultats stockés le {resultats['calcule_le']}.")
        else:
            st.caption("Aucun résultat stocké pour l'instant — lancez le calcul.")

    if clic and not disabled:
        with st.spinner("Calcul des prédictions en cours…"):
            class_names, val_y_true, val_y_pred, test_y_true, test_y_pred = fonction_calcul()
            sauvegarder_resultats(
                model_key, class_names, val_y_true, val_y_pred, test_y_true, test_y_pred
            )
            if summary_text:
                sauvegarder_summary(model_key, summary_text)
        resultats = charger_resultats_stockes(model_key)
        st.success("Résultats recalculés et stockés.")

    if not resultats:
        st.info("Cliquez sur « Calculer » pour lancer l'inférence et générer les résultats.")
        return

    tab_test, tab_val = st.tabs(["Test", "Validation"])
    with tab_val:
        evaluer(
            resultats["validation"]["y_true"], resultats["validation"]["y_pred"],
            resultats["class_names"], "Résultats — Validation",
        )
    with tab_test:
        evaluer(
            resultats["test"]["y_true"], resultats["test"]["y_pred"],
            resultats["class_names"], "Résultats — Test",
        )


# --------------------------------------------------------------------------- #
# Exécution
# --------------------------------------------------------------------------- #
st.sidebar.image(str(BASE_DIR / "intro.png"), width=400)
st.sidebar.title(":material/coronavirus: Sommaire")
pages=[
    "1.Introduction ",
    "2.Données & Visualisation",
    "3.Preprocessing",
    "4.Vers la modélisation",
    "5.Modèles de Machine Learning",
    "6.Modèles de Deep Learning",
    "7.Modèles & résultats",
    "8.Biais de source",
    "9.Impact du nombre de classes",
    "10.Analyse du meilleur modèle",
    "11.Conclusion",
    "12.Limites & perspectives",
    "13.Cas pratique"
]


# --------------------------------------------------------------------------- #
# Table de correspondance : ligne du tableau récapitulatif (diapo 5) -> diapo
# et modèle/variante à présélectionner. Seuls les modèles ayant une diapo
# dédiée sont mappés ; les autres lignes du tableau restent non cliquables.
# --------------------------------------------------------------------------- #
RECAP_ROW_TO_CIBLE = {
    "SVM (GridSearch best)": {"page": pages[5], "modele": "SVM"},
    "Regression logistique (balanced, OneVsRest)": {
        "page": pages[5], "modele": "Régression Logistique",
        "variante": "Balanced (avec pondération)",
    },
    "Regression logistique (sans ponderation, OneVsRest)": {
        "page": pages[5], "modele": "Régression Logistique",
        "variante": "Sans pondération",
    },
    "XGBoost (baseline, sans ponderation)": {
        "page": pages[5], "modele": "XGBoost", "variante": "Baseline",
    },
    "XGBoost (GridSearch best, balanced)": {
        "page": pages[5], "modele": "XGBoost", "variante": "GridSearch balanced",
    },
    "CNN 4 couches (tuned)": {"page": pages[6], "modele": "CNN 4 niveaux"},
}


#page=st.sidebar.radio("Aller vers", pages)

option_map = {p: f":material/coronavirus: {p}" for p in pages}

page = st.sidebar.pills(
    "",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
    default=pages[0],
    key="nav_page",
)
# --------------------------------------------------------------------------- #
# Introduction
# --------------------------------------------------------------------------- #
if page == pages[0]:
    st.subheader("Contexte & enjeux")
    st.write("""L'expansion rapide de l'épidémie de COVID-19 a très vite mis les systèmes de santé sous tension. Cet épisode a montré la nécessité d'obtenir un 
        diagnostic de manière instantanée et fiable. Celui-ci repose principalement sur le technique RT-PCR (Reverse Transcription Polymerase Chain Reaction), mais des études ont aussi mis en évidence certaines limites de cette technique.
        C'est pourquoi, l'imagerie médicale est apparue comme un outil complémentaire intéressant pour détecter les cas COVID.""")
    st.markdown(
            """
    **L'imagerie thoracique (radiographie) s'impose comme un outil complémentaire** : rapide,
    peu coûteuse, largement disponible — mais son interprétation reste complexe et demande
    l'expertise d'un radiologue.
    
    ### Objectif du projet
    Développer un modèle de **classification automatique** de radiographies pulmonaires capable
    de distinguer 4 catégories :
    - **COVID-19**
    - **Opacité pulmonaire** (Lung Opacity)
    - **Poumon normal**
    - **Pneumonie virale**
    
    ### Enjeu métier
    Un outil d'aide au **dépistage**, où le critère prioritaire n'est pas la performance globale
    mais bien la capacité à **ne pas manquer un cas COVID** (rappel/recall sur cette classe).
            """
        )
    st.write("""Notre projet propose de développer un modèle de classification automatique de radiographies pulmonaires capable de distinguer les cas COVID-19 des autres pathologies pulmonaires (pneumonie virale, opacité pulmonaire) et des poumons sains. Afin de répondre à cet objectif, nous disposions d'un jeu de données disponible ici :""")
    st.page_link("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database", label="COVID-19 Radiography Database")
    

# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
if page == pages[1] : 
    
    st.write("### Premier niveau d'analyse")

    st.write(f"Le jeu de données dont nous disposons contient 20 835 images réparties en 4 classes : Covid, Lung Opacity, Normal, Viral Pneumonia")

    st.image(str(BASE_DIR / "distribution.png"), caption="Répartition des classes", width=650)

    st.write("Les images COVID-19 proviennent de sources hétérogènes (PadChest, GitHub, SIRM, et autres dépôts publics), tandis que les classes Normal, Lung Opacity et Viral Pneumonia sont issues de bases de données uniques (RSNA, Kaggle).")

    st.write("#### Exemples d'images du dataset et les masques associés")
    st.image(str(PROJECT_ROOT / "reports" / "figures" / "exemples_images_masques.png"), width=750)

    st.write("### Distribution des pixels par classe")
    col1, col2 , col3, col4= st.columns(4)
    with col1:
        st.image(str(BASE_DIR / "hist_intensite_1.png"), caption="Sans masque")

    with col2:
        st.image(str(BASE_DIR / "hist_intensite_2_masques.png"), caption="Avec masque")

    with col3:
        st.image(str(BASE_DIR / "hist_intensite_3.png"), caption="Sans masque")

    with col4:
        st.image(str(BASE_DIR / "hist_intensite_4_masques.png"), caption="Avec masque")

    st.write("L'analyse de la luminosité des images montre des différences entre les différentes classes. La moyenne des pixels traduit le niveau global de luminosité d'une radiographie tandis que l'écart-type renseigne sur l'hétérogénité des nivveaux de gris. La classe Covid présente les intensités les plus élevées et ses valeurs sont plus dispersées tandis que les autres classes présentent des valeurs assez proches. L'apllication des masques modifient ces indicateurs en diminuant l'intensité lumineuse et en resserrant la distribution. Les différences entre les classes sont moins marquées.")


    st.write("### Analyse en composantes principales")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(str(BASE_DIR / "ACP_1.png"), width=420)

    with col2:
        st.image(str(BASE_DIR / "ACP_3.png"), width=420)

    with col3:
        st.image(str(BASE_DIR / "ACP_4.png"), width=420)

    st.write("Une analyse en composantes principales (ACP) sur le jeu de données est réalisée afin de visualiser la structure des données, repérer d'éventuels outliers et évaluer la séparation des différentes classes.")
    st.write("La forte superposition des classes suggère que les différences observées sont dues à des motifs radiologiques complexes et locaux. Cela justifie le recours a des modèles d'apprentissage capables d'extraire les caractéristiques discriminantes spécifique à chaque classe.")


    st.write("### Outliers")

    st.badge("Analyse des outliers sur le contraste => pas de problème au niveau de l'algorithme ", icon=":material/check:", color="green")

    st.write("Pour chaque classe, les images dont la distance quadratique moyenne (MSE) par rapport à l'image moyenne de leur classe dépasse un seuil ont été identifiées comme atypiques :")
    st.write("- Normal : 11 outliers (MSE > 2500)")
    st.write("- COVID : 34 outliers (MSE > 4 000)")
    st.write("- Lung Opacity : 20 outliers (MSE > 4 000)")
    st.write("- Viral Pneumonia : 32 outliers (MSE > 3 000)")
    st.write("Les distances sont calculées en travaillant sur une image redimensionnée à 50x50 pixels")

    st.badge("=> Après visualisation des outliers sur les écarts de dimension => pas de problème au niveau de l'algorithme ", icon=":material/check:", color="green")
    
    st.write("La détection des images floues a été réalisée selon le calcul de la variance du Laplacien (qui évalue la netteté de l'image). Les images présentant la variance la plus faible sont celles qui ont les contours les plus diffus")
    st.write("- Normal : 52 images floues")
    st.write("- COVID : 19  images floues")
    st.write("- Lung Opacity : 24 images floues")
    st.write("- Viral Pneumonia : 11 images floues")
    
    st.badge("=>Par le même mécanisme, les images dont la variance du laplacien n'atteignant pas un seuil ont été, cette fois, considérés comme floues et supprimées. ", icon=":material/check:", color="green")
             
    st.write("### Doublons")
    st.write("Pour chaque classe et pour chaque image, une distance quadratique moyenne moyenne (MSE) a été calculée entre chaque image :")
    st.badge("=>Par le même mécanisme,  le couple d'image dont la distance sera à inférieure à 10 seront considérer comme couple identique. La seconde image du couple sera supprimée ", icon=":material/check:", color="green")  

# --------------------------------------------------------------------------- #
# Feature Extraction
# --------------------------------------------------------------------------- #
if page == pages[2] :
    st.write("### Feature Extraction")
    
    features = pd.DataFrame({"Features":[
                "pixel_mean/pixel_std",
                "lum_interieur_masque / lum_exterieur_masque",
                "Surface_masque",
                "Variance laplacien"
            ],
            "Description":["Luminosité moyenne et dispersion des pixels sur l'image totale",
            "Luminosité moyenne dans et hors de la zone pulmonaire",
            "Proportion de pixels pulmonaires dans l'image",
            "Variance du Laplacien, indicateur de netteté de l'image"]
            })
    
    strategies_analyse = pd.DataFrame(
        {
            "Stratégie": [
                "Rééchantillonnage des features",
                "Class weights"
            ],
            "Description": [
                (
                    "Uniquement pour les modèles de Machine Learning entraînés sur les "
                    "features CSV. Un RandomOverSampler est appliqué au jeu "
                    "d'entraînement pour équilibrer les classes minoritaires. "
                    "Les jeux de validation et de test restent inchangés."
                ),
                (
                    "Un poids inversement proportionnel à la fréquence de chaque classe "
                    "est appliqué pendant l'entraînement afin d'accorder davantage "
                    "d'importance aux classes rares."
                )
            ]
        }
    )

    deduplication=pd.DataFrame({"Classe":["Covid", "Normal", "Lung Opacity", "Viral Pneumonia"],
                                    "Doublons":[1, 223, 0, 7],
                                    "outliers":[23, 55, 24, 0]})
    
    st.write("### Preprocessing")
    
    st.write("Pour répondre aux constats dressés lors de l'étape d'exploration, nous avons défini le pipeline de preprocessing ci-dessous.")
    
    st.image(str(BASE_DIR / "pipeline_preprocessing.png"), caption="Pipeline preprocessing", width=750)


    
    st.write("Cette étape comprend les processus de nettoyage, de traitement des images, de séparation et enfin d'augmantation sur le jeu d'entraînement uniquement.")
    
    st.write(
        "Cette stratégie a été adoptée pour fiabiliser le jeu de données "
        "(dé-duplication), supprimer le fond grâce aux masques, harmoniser les "
        "caractéristiques des images et augmenter la variabilité du jeu "
        "d'entraînement destiné aux modèles de Deep Learning."
    )
    
    st.write("L'étape d'exploration a permis de repérer 231 doublons et 102 outliers")
    st.table(deduplication)
    
    st.write("### Génération d'un dataset des caractéristiques des images")
    
    st.write("Un datase tabulaire a été généré en vue de l'entraînement des modèles de machine learning. Il contient l'ensemble des features des images traitées :")
    st.table(features)
    
    st.write("Le jeu de données ainsi constitué compte 20 835 images réparties sur les 4 classes.")
    
    st.write("### Gestion du déséquilibre des classes")

    st.write(
        "Deux stratégies distinctes ont été utilisées selon le type de modèle :"
    )

    st.table(strategies_analyse)

    st.caption(
        "Le rééchantillonnage concerne uniquement le fichier CSV d'entraînement "
        "utilisé par les modèles classiques de Machine Learning."
    )

    st.write("### Data augmentation pour les modèles de Deep Learning")

    st.markdown(
        """
La **data augmentation** augmente la variabilité des images grâce à plusieurs
transformations légères : rotations, translations, zooms et flips horizontaux.

Elle est appliquée **uniquement au jeu d'entraînement**. Une seule variante
augmentée est générée pour chaque image, puis enregistrée afin que tous les modèles
de Deep Learning soient comparés sur exactement le même jeu de données. Les jeux de
validation et de test restent inchangés.
        """
    )


    st.write("#### Exemple de data augmentation sur plusieurs images")

    st.caption(
        "Quatre radiographies différentes sont présentées avant et après "
        "l'application d'une seule transformation aléatoire. Les images augmentées "
        "ont ensuite été enregistrées afin que tous les modèles de Deep Learning "
        "soient entraînés sur exactement les mêmes données."
    )

    try:
        augmentation_dir = BASE_DIR / "data_augmentation_animation"

        augmentation_paths = [
            augmentation_dir / "image_01_originale.png",
            augmentation_dir / "image_01_augmentee.png",
            augmentation_dir / "image_02_originale.png",
            augmentation_dir / "image_02_augmentee.png",
            augmentation_dir / "image_03_originale.png",
            augmentation_dir / "image_03_augmentee.png",
            augmentation_dir / "image_04_originale.png",
            augmentation_dir / "image_04_augmentee.png",
        ]

        augmentation_images = [
            image_to_data_uri(str(image_path))
            for image_path in augmentation_paths
        ]

        # Lecture des paramètres exacts enregistrés lors de la génération
        # des variantes. Le CSV doit se trouver dans le même dossier que
        # les images augmentées.
        transformations_csv = (
            augmentation_dir / "transformations_data_augmentation.csv"
        )

        if not transformations_csv.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {transformations_csv}"
            )

        transformations_df = pd.read_csv(transformations_csv)

        colonnes_requises = {"fichier", "legende"}
        colonnes_manquantes = (
            colonnes_requises - set(transformations_df.columns)
        )

        if colonnes_manquantes:
            raise ValueError(
                "Colonnes manquantes dans le CSV des transformations : "
                + ", ".join(sorted(colonnes_manquantes))
            )

        # Réordonner les légendes selon l'ordre exact des images affichées.
        legendes_par_fichier = dict(
            zip(
                transformations_df["fichier"],
                transformations_df["legende"]
            )
        )

        augmentation_captions = [
            legendes_par_fichier[image_path.name]
            for image_path in augmentation_paths
        ]

        augmentation_images_js = ",\n".join(
            f'"{image_uri}"' for image_uri in augmentation_images
        )
        augmentation_captions_js = ",\n".join(
            f'"{caption}"' for caption in augmentation_captions
        )

        augmentation_html = f"""
        <style>
            .augmentation-wrapper {{
                max-width: 560px;
                margin: 0 auto;
                padding: 0 0 18px 0;
                font-family: Arial, sans-serif;
                text-align: center;
                box-sizing: border-box;
            }}

            .augmentation-stage {{
                position: relative;
                width: 100%;
                aspect-ratio: 1 / 1;
                overflow: hidden;
                border-radius: 12px;
                background: #0e1117;
            }}

            .augmentation-stage img {{
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
                opacity: 0;
                transform: scale(1.015);
                transition:
                    opacity 1.5s ease-in-out,
                    transform 1.5s ease-in-out;
                will-change: opacity, transform;
            }}

            .augmentation-stage img.active {{
                opacity: 1;
                transform: scale(1);
            }}

            .augmentation-caption {{
                min-height: 48px;
                margin-top: 12px;
                text-align: center;
                font-size: 15px;
                line-height: 1.45;
                color: #8b949e;
                opacity: 1;
                transition: opacity 0.55s ease-in-out;
            }}

            .augmentation-caption.fade {{
                opacity: 0;
            }}

            .augmentation-info {{
                margin-top: 8px;
                font-size: 13px;
                line-height: 1.4;
                color: #8b949e;
            }}

            .augmentation-launch {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 220px;
                margin-top: 14px;
                margin-bottom: 8px;
                border: none;
                border-radius: 10px;
                background: #ff4b4b;
                color: white;
                padding: 10px 18px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                line-height: 1.2;
                white-space: nowrap;
                box-sizing: border-box;
            }}

            .augmentation-launch:hover {{
                filter: brightness(0.96);
            }}

            .augmentation-launch:disabled {{
                cursor: default;
                opacity: 0.75;
            }}
        </style>

        <div class="augmentation-wrapper">
            <div class="augmentation-stage" id="augmentation-stage"></div>

            <div class="augmentation-caption" id="augmentation-caption">
                Radiographie 1 — image originale.
            </div>

            <div class="augmentation-info">
                Chaque radiographie possède une seule variante augmentée enregistrée.
                Les légendes indiquent les paramètres exacts appliqués.
            </div>

            <button type="button" class="augmentation-launch" id="augmentation-launch">
                ▶ Lancer l'animation
            </button>
        </div>

        <script>
            const augmentationSources = [
                {augmentation_images_js}
            ];

            const augmentationCaptions = [
                {augmentation_captions_js}
            ];

            const augmentationStage = document.getElementById("augmentation-stage");
            const augmentationCaption = document.getElementById("augmentation-caption");
            const augmentationButton = document.getElementById("augmentation-launch");

            const augmentationElements = augmentationSources.map((source, index) => {{
                const image = document.createElement("img");
                image.src = source;
                image.alt = augmentationCaptions[index];
                if (index === 0) {{
                    image.classList.add("active");
                }}
                augmentationStage.appendChild(image);
                return image;
            }});

            let augmentationTimers = [];

            function clearAugmentationTimers() {{
                augmentationTimers.forEach(timer => clearTimeout(timer));
                augmentationTimers = [];
            }}

            function updateAugmentationCaption(index) {{
                augmentationCaption.classList.add("fade");

                augmentationTimers.push(setTimeout(() => {{
                    augmentationCaption.textContent = augmentationCaptions[index];
                    augmentationCaption.classList.remove("fade");
                }}, 280));
            }}

            function showAugmentationStep(index) {{
                augmentationElements.forEach((image, imageIndex) => {{
                    image.classList.toggle("active", imageIndex === index);
                }});
                updateAugmentationCaption(index);
            }}

            function launchAugmentation() {{
                clearAugmentationTimers();
                augmentationButton.disabled = true;
                augmentationButton.textContent = "Animation en cours…";

                augmentationElements.forEach((image, index) => {{
                    image.classList.toggle("active", index === 0);
                }});
                augmentationCaption.textContent = augmentationCaptions[0];
                augmentationCaption.classList.remove("fade");

                const displayDuration = 2100;

                for (let index = 1; index < augmentationElements.length; index += 1) {{
                    augmentationTimers.push(setTimeout(() => {{
                        showAugmentationStep(index);
                    }}, displayDuration * index));
                }}

                augmentationTimers.push(setTimeout(() => {{
                    augmentationButton.disabled = false;
                    augmentationButton.textContent = "▶ Relancer l'animation";
                }}, displayDuration * augmentationElements.length));
            }}

            augmentationButton.addEventListener("click", launchAugmentation);
        </script>
        """

        components.html(augmentation_html, height=760, scrolling=False)

    except (FileNotFoundError, ValueError, KeyError) as e:
        st.warning(
            f"{e}. Vérifiez que le dossier data_augmentation_animation "
            "contient les huit images (quatre originales et quatre augmentées) "
            "ainsi que le fichier transformations_data_augmentation.csv."
        )


# --------------------------------------------------------------------------- #
# Vers la modélisation
# --------------------------------------------------------------------------- #
if page == pages[3] :
    st.write("### Preprocessing")

    st.markdown(
        """
- **Segmentation pulmonaire** : application du masque fourni avec le dataset (généré par
  un U-Net par les auteurs), pour isoler la zone pulmonaire et limiter le biais de source
  concentré dans le fond des images.
- **Normalisation de la luminosité** : égalisation d'histogramme puis CLAHE (contraste
  adaptatif local), pour homogénéiser les images entre sources.
- **Dédoublonnage** : détection des quasi-doublons par distance MSE à l'image moyenne de
  la classe.
- **Split stratifié** 80/10/10 (train / validation / test), graine fixée.
- **Data augmentation** (train uniquement) : rotation, translation, zoom, flip horizontal —
  transformations légères pour ne pas altérer les caractéristiques médicales des lésions.
        """
    )
    st.write("#### Exemple de preprocessing sur une image COVID")

    st.caption(
        "L'animation illustre l'application du masque pulmonaire, puis "
        "l'amélioration locale du contraste par CLAHE."
    )

    try:
        image_brute = image_to_data_uri(str(BASE_DIR / "preprocessing_1_brute.png"))
        image_masque = image_to_data_uri(str(BASE_DIR / "preprocessing_2_masque.png"))
        image_clahe = image_to_data_uri(str(BASE_DIR / "preprocessing_3_clahe.png"))

        # Composant HTML compact avec transition progressive entre les étapes.
        animation_html = f"""
        <style>
            .preprocess-wrapper {{
                max-width: 580px;
                margin: 0 auto;
                padding: 0 0 18px 0;
                font-family: Arial, sans-serif;
                text-align: center;
                box-sizing: border-box;
            }}

            .preprocess-stage {{
                position: relative;
                width: 100%;
                aspect-ratio: 1.75 / 1;
                overflow: hidden;
                border-radius: 12px;
                background: #0e1117;
            }}

            .preprocess-stage img {{
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
                opacity: 0;
                transform: scale(1.015);
                transition:
                    opacity 1.8s ease-in-out,
                    transform 1.8s ease-in-out;
                will-change: opacity, transform;
            }}

            .preprocess-stage img.active {{
                opacity: 1;
                transform: scale(1);
            }}

            .preprocess-caption {{
                min-height: 46px;
                margin-top: 12px;
                text-align: center;
                font-size: 15px;
                line-height: 1.45;
                color: #8b949e;
                opacity: 1;
                transition: opacity 0.6s ease-in-out;
            }}

            .preprocess-caption.fade {{
                opacity: 0;
            }}

            .preprocess-launch {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 210px;
                margin-top: 14px;
                margin-bottom: 8px;
                border: none;
                border-radius: 10px;
                background: #ff4b4b;
                color: white;
                padding: 10px 18px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                line-height: 1.2;
                white-space: nowrap;
                box-sizing: border-box;
            }}

            .preprocess-launch:hover {{
                filter: brightness(0.96);
            }}

            .preprocess-launch:disabled {{
                cursor: default;
                opacity: 0.75;
            }}
        </style>

        <div class="preprocess-wrapper">
            <div class="preprocess-stage">
                <img src="{image_brute}" alt="Radiographie brute" class="active">
                <img src="{image_masque}" alt="Radiographie après application du masque">
                <img src="{image_clahe}" alt="Radiographie après CLAHE et masque">
            </div>

            <div class="preprocess-caption" id="preprocess-caption">
                Image brute — radiographie avant traitement.
            </div>

            <button type="button" class="preprocess-launch" id="preprocess-launch">
                ▶ Lancer l'animation
            </button>
        </div>

        <script>
            const images = Array.from(document.querySelectorAll(".preprocess-stage img"));
            const caption = document.getElementById("preprocess-caption");
            const launchButton = document.getElementById("preprocess-launch");

            const captions = [
                "Image brute — radiographie avant traitement.",
                "Application du masque — seules les zones pulmonaires sont conservées.",
                "CLAHE + masque — le contraste local est progressivement renforcé."
            ];

            let timers = [];

            function clearTimers() {{
                timers.forEach(timer => clearTimeout(timer));
                timers = [];
            }}

            function updateCaption(index) {{
                caption.classList.add("fade");

                timers.push(setTimeout(() => {{
                    caption.textContent = captions[index];
                    caption.classList.remove("fade");
                }}, 320));
            }}

            function showStep(index) {{
                images.forEach((image, i) => {{
                    image.classList.toggle("active", i === index);
                }});
                updateCaption(index);
            }}

            function launchAnimation() {{
                clearTimers();
                launchButton.disabled = true;
                launchButton.textContent = "Animation en cours…";

                images.forEach((image, i) => {{
                    image.classList.toggle("active", i === 0);
                }});
                caption.textContent = captions[0];
                caption.classList.remove("fade");

                timers.push(setTimeout(() => {{
                    showStep(1);
                }}, 2400));

                timers.push(setTimeout(() => {{
                    showStep(2);
                }}, 5200));

                timers.push(setTimeout(() => {{
                    launchButton.disabled = false;
                    launchButton.textContent = "▶ Relancer l'animation";
                }}, 8000));
            }}

            launchButton.addEventListener("click", launchAnimation);
        </script>
        """

        components.html(animation_html, height=485, scrolling=False)

    except FileNotFoundError as e:
        st.warning(
            f"{e}. Les trois images de l'animation doivent être placées "
            "dans le même dossier que app.py."
        )

    st.write("### Vers la modélisation")
    st.markdown(
        """
Une fois les images préparées, nous avons comparé deux familles d'approches pour la classification :

**Machine Learning classique** — entraîné sur les features tabulaires extraites des images
(luminosité, texture, surface du masque…). Algorithmes testés : SVM, Random Forest, XGBoost,
KNN, Régression logistique ; nous détaillerons le **SVM**, notre meilleur modèle ML.

**Deep Learning** — entraîné directement sur les images brutes, sans extraction manuelle de
features. Architectures testées : CNN (plusieurs profondeurs), DenseNet, VGG16, Inception V3,
ResNet50, ResNet101V2, CovidNet, EfficientNet ; nous détaillerons le **CNN à 4 niveaux**, le
modèle retenu.

Métrique de comparaison retenue : **F1-macro**, pour ne pas favoriser la classe majoritaire
et bien évaluer la détection de la classe COVID (minoritaire).
        """
    )


# --------------------------------------------------------------------------- #
# Modèles & résultats
# --------------------------------------------------------------------------- #
if page == pages[6] :
    st.write("### Modèles entraînés & résultats")

    st.markdown(
        """
Deux familles de modèles comparées : **Machine Learning classique** (sur descripteurs
tabulaires) et **Deep Learning** (directement sur les images). Métrique de référence :
**F1-macro**, qui ne se laisse pas tirer vers le haut par la classe majoritaire et
pénalise un modèle qui rate une classe minoritaire comme COVID.

Un **DummyClassifier** (aucun apprentissage réel) sert de plancher de référence :
F1-macro de 0,16 à 0,25 selon la stratégie — tout modèle utile doit largement le dépasser.
        """
    )
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "f1_macro_comparaison_2.png"), caption="F1-macro — comparaison Machine Learning vs Deep Learning (test)", width=1400)

    st.write("#### Tableau récapitulatif final")
    st.caption(
        "Cliquez sur une ligne pour ouvrir la diapo détaillée du modèle, quand elle "
        "existe (SVM, Régression Logistique, XGBoost, CNN 4 couches tuned)."
    )
    recap = pd.read_csv(str(PROJECT_ROOT / "models" / "model_comparison_recap_final.csv"))
    selection_recap = st.dataframe(
        recap,
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key="recap_table",
    )

    lignes_selectionnees = selection_recap.selection.rows if selection_recap else []
    if lignes_selectionnees:
        nom_modele = recap.iloc[lignes_selectionnees[0]]["modele"]
        cible = RECAP_ROW_TO_CIBLE.get(nom_modele)
        if cible is None:
            st.info(f"Pas de diapo détaillée disponible pour « {nom_modele} ».")
        else:
            st.session_state["nav_page"] = cible["page"]
            if "modele" in cible:
                cle_modele = "ml_modele_select" if cible["page"] == pages[5] else "dl_modele_select"
                st.session_state[cle_modele] = cible["modele"]
            if "variante" in cible:
                cle_variante = (
                    "logreg_variante_select" if cible["modele"] == "Régression Logistique"
                    else "xgboost_variante_select"
                )
                st.session_state[cle_variante] = cible["variante"]
            st.rerun()


# --------------------------------------------------------------------------- #
# Modèles Machine Learning (SVM, Régression Logistique, XGBoost)
# --------------------------------------------------------------------------- #
if page == pages[4] :
    st.write("### Modèles de Machine Learning")

    modele_ml = st.pills(
        "Modèle",
        ["SVM", "Régression Logistique", "XGBoost", "Random Forest"],
        selection_mode="single",
        default="SVM",
        key="ml_modele_select",
    )

    if modele_ml == "SVM":
        
        svm_ml = st.pills(
        "variante",
        ["normal", "weighted"],
        selection_mode="single",
        default="weighted",
        key="svm_mode_select",
        )
        
        if svm_ml == "normal":
            st.write("#### SVM : Best gridsearch ")
            model_SVM_path_select = model_SVM_path
            scaler_SVM_path_select = scaler_SVM_path
        if svm_ml == "weighted":
            st.write("#### SVM : Best gridsearch + weighted")
            model_SVM_path_select = model_SVM_Weighted_path
            scaler_SVM_path_select = scaler_SVM_Weighted_path

        def calculer_svm():
            try:
                model_loaded = get_model_SVM(model_SVM_path_select)
            except Exception as e:
                st.error(f"Impossible de charger le modèle SVM: {e}")
                st.stop()

            try:
                scaler_loaded = get_scaler_SVM(scaler_SVM_path_select)
            except Exception as e:
                st.error(f"Impossible de charger le scaler SVM : {e}")
                st.stop()

            try:
                df_test = pd.read_csv(csv_test)
                df_validation = pd.read_csv(csv_validation)
            except Exception as e:
                st.error(f"Impossible de charger les fichiers de features : {e}")
                st.stop()

            X_test = df_test.drop(['filename', 'classe'], axis=1)
            y_test = df_test['classe']

            X_val = df_validation.drop(['filename', 'classe'], axis=1)
            y_val = df_validation['classe']

            X_test_scaled = scaler_loaded.transform(X_test)
            X_val_scaled = scaler_loaded.transform(X_val)

            test_pred_class = model_loaded.predict(X_test_scaled)
            val_pred_class = model_loaded.predict(X_val_scaled)

            class_names = sorted(y_test.unique())

            return class_names, y_val, val_pred_class, y_test, test_pred_class

        # Une clé de stockage par variante (normal/weighted), pour ne pas
        # écraser les résultats de l'une avec ceux de l'autre.
        afficher_section_avec_calcul(f"SVM_{svm_ml}", calculer_svm)

    elif modele_ml == "Régression Logistique":
        st.write("#### Régression Logistique")
        st.caption(
            "Résultats précalculés (issus du rapport de comparaison), identiques quelle "
            "que soit la machine ou la version des librairies utilisées le jour J. "
            "Seul le jeu de test a été évalué pour ces deux variantes."
        )

        variante_logreg = st.pills(
            "Variante",
            list(LOGREG_RESULTS.keys()),
            selection_mode="single",
            default="Balanced (avec pondération)",
            key="logreg_variante_select",
        )

        evaluer_depuis_matrice(
            LOGREG_RESULTS[variante_logreg]["Test"],
            FROZEN_CLASS_NAMES,
            f"Résultats — Test ({variante_logreg})",
        )

    elif modele_ml == "XGBoost":
        st.write("#### XGBoost")
        st.caption(
            "Résultats précalculés (issus du rapport de comparaison), identiques quelle "
            "que soit la machine ou la version des librairies utilisées le jour J."
        )

        variante_xgb = st.pills(
            "Variante",
            list(XGBOOST_RESULTS.keys()),
            selection_mode="single",
            default="Baseline",
            key="xgboost_variante_select",
        )

        tab_test, tab_val = st.tabs(["Test", "Validation"])
        with tab_val:
            evaluer_depuis_matrice(
                XGBOOST_RESULTS[variante_xgb]["Validation"],
                FROZEN_CLASS_NAMES,
                "Résultats — Validation",
            )
        with tab_test:
            evaluer_depuis_matrice(
                XGBOOST_RESULTS[variante_xgb]["Test"],
                FROZEN_CLASS_NAMES,
                "Résultats — Test",
            )

    elif modele_ml == "Random Forest":
        
        st.write("#### Random Forest")    
   
        def calculer_Random_Forest():
            try:
                model_loaded = get_model_SVM(model_random_forest_Weighted_path)
            except Exception as e:
                st.error(f"Impossible de charger le modèle Random Forrest: {e}")
                st.stop()
    
            
            try:
                df_test = pd.read_csv(csv_test)
                df_validation = pd.read_csv(csv_validation)
            except Exception as e:
                st.error(f"Impossible de charger les fichiers de features : {e}")
                st.stop()
    
            X_test = df_test.drop(['filename', 'classe'], axis=1)
            y_test = df_test['classe']
    
            X_val = df_validation.drop(['filename', 'classe'], axis=1)
            y_val = df_validation['classe']
    
    
            test_pred_class = model_loaded.predict(X_test)
            val_pred_class = model_loaded.predict(X_val)
    
            class_names = sorted(y_test.unique())
    
            return class_names, y_val, val_pred_class, y_test, test_pred_class
    
            # Une clé de stockage par variante (normal/weighted), pour ne pas
            # écraser les résultats de l'une avec ceux de l'autre.
        afficher_section_avec_calcul("RForrest", calculer_Random_Forest)

# --------------------------------------------------------------------------- #
# Modèles Deep Learning (CNN)
# --------------------------------------------------------------------------- #
if page == pages[5] :
    st.write("### Modèles de Deep Learning")

    modele_dl = st.pills(
        "Modèle",
        [
            "CNN 1 niveaux", "CNN 2 niveaux", "CNN 3 niveaux", "CNN 4 niveaux","CNN 4 niveaux tuned",
            "CNN 5 niveaux", "CNN 6 niveaux", "VGG16", "InceptionV3", "DenseNet",
        ],
        selection_mode="single",
        default="CNN 4 niveaux tuned",
        key="dl_modele_select",
    )

    # Par défaut : CNN maison en niveaux de gris, pas de preprocessing dédié.
    # Les modèles de transfer learning (VGG16, InceptionV3, DenseNet)
    # écrasent ces valeurs avec leurs propres taille d'image, mode couleur
    # (RVB) et fonction de preprocessing.
    color_mode = "grayscale"
    img_h = img_w = size_img
    preprocess_fn = None

    # `model_loaded` reste à None si le fichier .keras est introuvable/corrompu :
    # on n'interrompt plus le script (plus de st.stop()) pour permettre
    # l'affichage du résumé stocké et le grisage du bouton Recalculer.
    model_loaded = None

    if modele_dl == "CNN 6 niveaux":
        st.write("#### CNN 6 niveaux")
        st.image(courbe_CNN1024_path, width=600)
        try:
            model_loaded = get_model_CNN1024()
        except Exception as e:
            st.warning(f"⚠️ Modèle CNN1024 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")


    elif modele_dl == "CNN 5 niveaux":
        st.write("#### CNN 5 niveaux")
        st.image(courbe_CNN512_path, width=600)
        try:
            model_loaded = get_model_CNN512()
        except Exception as e:
            st.warning(f"⚠️ Modèle CNN512 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")

    elif modele_dl == "CNN 4 niveaux":
        st.write("#### CNN 4 niveaux")
        st.image(courbe__CNN256_path, width=600)
        try:
            model_loaded = get_model_CNN256()
        except Exception as e:
            st.warning(f"⚠️ Modèle CNN256 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")
            
    elif modele_dl == "CNN 4 niveaux tuned":
        st.write("#### CNN 4 niveaux tuned")
        st.caption(
        "recherche de la meilleure solution avec keras_tuner sur hyperparamètres : dropout / dense / L2 / learning_rate"
        )
        st.image(courbe_CNN256_tuned_path, width=600)
        try:
            model_loaded = get_model_CNN256_tuned()
        except Exception as e:
            st.warning(f"⚠️ Modèle CNN256 tuned indisponible ({e}). Affichage à partir des données stockées, si disponibles.")
    
    elif modele_dl == "CNN 3 niveaux":
        st.write("#### CNN 3 niveaux")
        st.image(courbe__CNN128_path, width=600)
        try:
            model_loaded = get_model_CNN128()
        except Exception as e:
            st.warning(f"⚠️ Modèle CNN128 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")
            
    elif modele_dl == "CNN 2 niveaux":
            st.write("#### CNN 2 niveaux")
            st.image(courbe_CNN64_path, width=600)
            try:
                model_loaded = get_model_CNN64()
            except Exception as e:
                st.warning(f"⚠️ Modèle CNN64 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")
                
    elif modele_dl == "CNN 1 niveaux":
                st.write("#### CNN 1 niveaux")
                st.image(courbe_CNN32_path, width=600)
                try:
                    model_loaded = get_model_CNN32()
                except Exception as e:
                    st.warning(f"⚠️ Modèle CNN32 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")

    elif modele_dl == "VGG16":
        st.write("#### VGG16 (transfer learning)")
        st.caption(
                " stratégie de transfert learning : Fine-Tuning partiel de la dernière couche profonde sur 20 epoch"
            )
        st.image(courbe_VGG16_path, width=600)

        try:
            model_loaded = get_model_VGG16_finetuned()
        except Exception as e:
            st.warning(f"⚠️ Modèle VGG16 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")

        color_mode = "rgb"
        img_h = img_w = 224
        preprocess_fn = vgg16_preprocess_input

    elif modele_dl == "InceptionV3":
        st.write("#### InceptionV3 (transfer learning)")
        st.caption(
                " stratégie de transfert learning : Fine-Tuning partiel des derniers 20% couche profonde sur 20 epoch"
            )
        st.image(courbe_Inceptionv3_path, width=600)
        try:
            model_loaded = get_model_InceptionV3_finetuned()
        except Exception as e:
            st.warning(f"⚠️ Modèle InceptionV3 indisponible ({e}). Affichage à partir des données stockées, si disponibles.")

        color_mode = "rgb"
        img_h = img_w = 299
        preprocess_fn = inceptionv3_preprocess_input

    elif modele_dl == "DenseNet":
        st.write("#### DenseNet (transfer learning)")
        st.caption(
                " stratégie de transfert learning : Fine-Tuning partiel de la moitié des couches sur 30 epoch"
            )       
        st.image(courbe_DenseNet_path, width=600)
       
        try:
            model_loaded = get_model_DenseNet()
        except Exception as e:
            st.warning(f"⚠️ Modèle DenseNet indisponible ({e}). Affichage à partir des données stockées, si disponibles.")

        color_mode = "rgb"
        img_h = img_w = 299
        # Le modèle densenet.keras embarque déjà sa propre normalisation
        # (true_divide / add / true_divide_1 juste après l'Input, cf. notebook
        # d'entraînement où densenet_preprocess_input n'est jamais appliqué
        # manuellement). Appliquer preprocess_fn ici double le preprocessing
        # et corrompt les prédictions.
        preprocess_fn = None

    # ----------------------------------------------------------------------- #
    # Résumé du modèle : lu en direct sur le modèle chargé s'il est
    # disponible (et alors stocké en dur pour servir de secours plus tard),
    # sinon relu depuis le fichier stocké par un précédent calcul — sans
    # jamais lever d'erreur si le modèle .keras est absent/corrompu.
    # ----------------------------------------------------------------------- #
    summary_text = None
    with st.expander("📋 Résumé du modèle"):
        if model_loaded is not None:
            summary_lines = []
            model_loaded.summary(print_fn=lambda x: summary_lines.append(x))
            summary_text = "\n".join(summary_lines)
            st.code(summary_text)
        else:
            summary_text = charger_summary_stocke(modele_dl)
            if summary_text:
                st.info("Modèle indisponible : résumé affiché depuis la dernière sauvegarde.")
                st.code(summary_text)
            else:
                st.info("Modèle indisponible et aucun résumé n'a encore été stocké pour ce modèle.")

    def calculer_dl():
        try:
            val_ds, class_names = get_dataset_dl(
                val_dir, img_h=img_h, img_w=img_w, batch_size=32,
                color_mode=color_mode, _preprocess_fn=preprocess_fn,
            )
            test_ds, class_names = get_dataset_dl(
                test_dir, img_h=img_h, img_w=img_w, batch_size=32,
                color_mode=color_mode, _preprocess_fn=preprocess_fn,
            )
        except Exception as e:
            st.error(f"Impossible de charger les jeux de données : {e}")
            st.stop()

        test_pred = model_loaded.predict(test_ds)
        test_pred_class = test_pred.argmax(axis=1)
        y_true_test_class = np.concatenate([labels for _, labels in test_ds], axis=0)

        val_pred = model_loaded.predict(val_ds)
        val_pred_class = val_pred.argmax(axis=1)
        y_true_val_class = np.concatenate([labels for _, labels in val_ds], axis=0)

        return class_names, y_true_val_class, val_pred_class, y_true_test_class, test_pred_class

    # Une clé de stockage par modèle sélectionné dans les pills. Le bouton
    # Recalculer est grisé si le modèle n'a pas pu être chargé, et le résumé
    # affiché est stocké en dur à chaque recalcul réussi.
    afficher_section_avec_calcul(
        modele_dl, calculer_dl,
        disabled=(model_loaded is None),
        summary_text=summary_text,
    )


# --------------------------------------------------------------------------- #
# Biais de source
# --------------------------------------------------------------------------- #
if page == pages[7] :
    st.write("### Biais de source dans les modèles")
    st.markdown(
        """
Les images COVID proviennent de 4 sources hétérogènes (SIRM, Github, PadChest, Eurorad).
L'EDA seule ne suffit pas : on a vérifié si les **modèles** performent différemment selon
la source de l'image, pas seulement si les images elles-mêmes diffèrent statistiquement.
        """
    )
    st.image(str(PROJECT_ROOT / "reports" / "figures" / "source_bias_covid_recall.png"), width=750)
    st.caption(
        "Rappel COVID très variable selon la source pour les modèles Machine Learning "
        "(à nuancer : Eurorad n=30 et SIRM n=14 sont de petits effectifs)."
    )


# --------------------------------------------------------------------------- #
# Impact du nombre de classes
# --------------------------------------------------------------------------- #
if page == pages[8] :
    st.write("### Impact du nombre de classes : 3 classes vs 4 classes")

    st.markdown(
        """
Le test post-hoc de Dunn, corrigé par Bonferroni, montre que les classes
**Lung Opacity** et **Viral Pneumonia** sont statistiquement équivalentes
pour l'intensité moyenne des pixels (**p = 1**).

Ce résultat nous a conduits à tester leur fusion dans une classe unique.
        """
    )

    # Tableau volontairement plus compact que les graphiques de résultats.
    col_gauche, col_dunn, col_droite = st.columns([1, 2.2, 1])
    with col_dunn:
        st.image(
            str(BASE_DIR / "impact_classe_dunn.png"),
            caption=(
                "Test de Dunn sur l'intensité moyenne : aucune différence "
                "significative entre Lung Opacity et Viral Pneumonia."
            ),
            width="stretch"
        )

    st.markdown(
        """
Nous avons ensuite comparé deux configurations :

- **4 classes** : COVID, Normal, Lung Opacity et Viral Pneumonia ;
- **3 classes** : COVID, Normal et fusion des deux classes pathologiques.
        """
    )

    tab_global, tab_covid = st.tabs(
        ["F1-macro global", "F1 de la classe COVID"]
    )

    with tab_global:
        st.image(
            str(BASE_DIR / "impact_classe_f1global.png"),
            caption="Comparaison du F1-macro global en 3 et 4 classes.",
            width=750
        )
        st.markdown(
            """
**À retenir :** les écarts sont faibles, mais les modèles de Deep Learning
sont globalement légèrement meilleurs en **4 classes**.
            """
        )

    with tab_covid:
        st.image(
            str(BASE_DIR / "impact_classe_f1covid.png"),
            caption="Comparaison du F1-score COVID en 3 et 4 classes.",
            width=750
        )
        st.markdown(
            """
**À retenir :** la fusion n'améliore pas systématiquement la détection du COVID.
Les performances restent proches selon les modèles.
            """
        )

    st.success(
        """
        **Choix retenu : 4 classes**, pour conserver une information plus détaillée
        sans perte de performance.
        """
    )


# --------------------------------------------------------------------------- #
# Analyse du meilleur modèle
# --------------------------------------------------------------------------- #
if page == pages[9] :
    st.write("### Analyse du modèle retenu — CNN 4 couches (tuned)")

    st.markdown(
        """
**Modèle retenu : CNN 4 couches (tuned)** — F1-macro 0,90 (légèrement en retrait sur ce
seul critère face à CNN 5 couches et DenseNet, tous deux à 0,92), mais choisi pour deux
raisons prioritaires pour un outil de dépistage :
- **Meilleur rappel COVID de tous les modèles testés : 93,5 %**
- **Surapprentissage plus contenu** (écart train/validation +0,06, contre +0,08 pour le
  CNN 5 couches)
        """
    )
    st.caption("Survolez une image avec la souris pour l'agrandir.")

    try:
        image_cm_uri = image_to_data_uri(str(PROJECT_ROOT / "reports" / "figures" / "cm_deep_learning_normalized.png"))
        image_pr_uri = image_to_data_uri(str(PROJECT_ROOT / "reports" / "figures" / "rpt_pr_curves_dl.png"))

        zoom_html = f"""
        <style>
            .zoom-row {{
                display: flex;
                gap: 80px;
                justify-content: center;
                align-items: flex-start;
                font-family: Arial, sans-serif;
                padding: 20px 80px 320px 80px;
                box-sizing: border-box;
            }}
            .zoom-item {{
                flex: 1;
                max-width: 46%;
                text-align: center;
            }}
            .zoom-frame {{
                overflow: visible;
            }}
            .zoom-frame img {{
                height: 380px;
                width: auto;
                max-width: 100%;
                display: block;
                margin: 0 auto;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25);
                transition: transform 0.35s ease, box-shadow 0.35s ease;
                cursor: zoom-in;
                position: relative;
                z-index: 1;
            }}
            /* Image de gauche : le centre agrandi se décale de 20% vers la droite */
            .zoom-item:first-child .zoom-frame img {{
                transform-origin: 21% top;
            }}
            /* Image de droite : le centre agrandi se décale de 20% vers la gauche */
            .zoom-item:last-child .zoom-frame img {{
                transform-origin: 79% top;
            }}
            .zoom-frame img:hover {{
                transform: scale(1.85);
                box-shadow: 0 14px 34px rgba(0,0,0,0.55);
                z-index: 50;
            }}
            .zoom-caption {{
                margin-top: 12px;
                font-size: 14px;
                color: #8b949e;
            }}
        </style>

        <div class="zoom-row">
            <div class="zoom-item">
                <div class="zoom-frame">
                    <img src="{image_cm_uri}" alt="Matrices de confusion normalisées — Deep Learning">
                </div>
                <div class="zoom-caption">Matrices de confusion normalisées — Deep Learning</div>
            </div>
            <div class="zoom-item">
                <div class="zoom-frame">
                    <img src="{image_pr_uri}" alt="Courbes Précision-Rappel — Deep Learning">
                </div>
                <div class="zoom-caption">Courbes Précision-Rappel — Deep Learning</div>
            </div>
        </div>
        """
        components.html(zoom_html, height=750, scrolling=False)

    except (FileNotFoundError, ValueError) as e:
        st.warning(f"{e}")
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(PROJECT_ROOT / "reports" / "figures" / "cm_deep_learning_normalized.png"), caption="Matrices de confusion normalisées — Deep Learning", width=650)
        with col2:
            st.image(str(PROJECT_ROOT / "reports" / "figures" / "rpt_pr_curves_dl.png"), caption="Courbes Précision-Rappel — Deep Learning", width=650)

    st.write("#### Grad-CAM — où le modèle « regarde »")
    st.image(str(PROJECT_ROOT / "reports" / "figures" / "gradcam_planche.png"), caption="Cartes d'activation Grad-CAM par modèle", width=750)


# --------------------------------------------------------------------------- #
# Conclusion
# --------------------------------------------------------------------------- #
if page == pages[10] :
    st.write("### Conclusion")

    st.markdown(
        """
Ce projet visait à classer automatiquement des radiographies thoraciques en 4 catégories.
La démarche a suivi trois grandes étapes : exploration approfondie, pré-processing rigoureux,
puis modélisation comparant plusieurs familles d'algorithmes.

**Hiérarchie claire des résultats :**
- Les baselines Machine Learning plafonnent à un F1-macro de 0,61 (SVM, meilleur score) et
  détectent mal la classe COVID.
- Les modèles Deep Learning franchissent nettement ce plafond (F1-macro jusqu'à 0,92) et
  améliorent fortement la détection COVID.

**Modèle retenu : CNN 4 couches (tuned).** Trois modèles partagent le meilleur F1-macro
(0,92 : CNN 4 couches, CNN 5 couches, DenseNet), mais le CNN 4 couches (tuned) se distingue
par le **meilleur rappel COVID (93,5 %)** — le critère prioritaire pour un outil de dépistage
— et un surapprentissage mieux maîtrisé que le CNN 5 couches. DenseNet, à score égal, n'a
pas été évalué sur ces deux critères dans ce rapport.

Ce compromis précision/rappel devra être calibré selon le contexte clinique visé.
        """
    )


# --------------------------------------------------------------------------- #
# Limites & perspectives
# --------------------------------------------------------------------------- #
if page == pages[11] :
    st.write("### Limites & perspectives")

    st.markdown(
        """
- **Biais de source** : testé sur les modèles ML classiques (rappel COVID variable selon
  la source), pas encore étendu aux modèles Deep Learning.
- **Fuite train/test potentielle** : rien ne garantit qu'un même patient n'apparaît pas à
  la fois en train et en test (dé-duplication actuelle basée sur la similarité d'image, pas
  sur l'identité patient).
- **Calibration du seuil de décision** : le compromis précision/rappel sur COVID n'a pas
  été calibré pour un contexte clinique précis.
- **Scoring GridSearch** : confirmé en F1-macro pour SVM et KNN ; à vérifier pour Random
  Forest et XGBoost.
- **Transfer learning** : DenseNet, à performance égale au meilleur CNN maison, mériterait
  une analyse plus poussée (rappel COVID, surapprentissage) avant d'envisager de le retenir.
- **Transfer learning** :  sur les modèles, la profondeur du dégel n'a pas fait l'objet d'une grande optimisation. Cela était couteux.
- **CNN tuned** :  Le temps de recherche a été limité en terme d'epoch et de temps (1 nuit)
- **Validation externe** : le modèle retenu n'a pas été testé sur des données d'une autre
  origine que ce dataset.
  **Validation externe** : le modèle retenu n'a pas été testé sur des données d'une autre
  origine que ce dataset.
        """
    )

    st.write("### Expertise sur les détections gradcam")
    st.caption("Analyse de détections correctes")
    
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "gradcamCovidBonneDetection.png"))
    st.markdown("""
            **COVID**: image de gauche et centrale : activation concentrée en amas irréguliers -zone médio-basale-, plutôt multifocale/ par touches, cohérente avec des opacités en verre dépoli typiques du COVID. Le modèle trouve certains éléments intéressants sur le bas des poumons par contre il se trompe complètement sur la partie haute des poumons où il confond le flou lié à l’infection avec le flou lié à la superposition de la clavicule / omoplate / côte épaisse. 
    La troisième image. L’activation se fait essentiellement sur la partie haute des poumons et c’est une erreur. La bonne détection s’est faite sur de mauvais critères"""
    )
    
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "gradcamLungOpacityBonneDetection.png"))
    st.markdown("""
                **Lung_Opacity**: zones plus centrales et linéaires. Le modèle trouve certains éléments intéressants comme des opacités mal-définies par contre il se trompe complètement sur la partie haute des poumons où il confond le flou lié à l’infection avec le flou lié à la superposition de la clavicule / omoplate / côte épaisse. On retrouve des activations sur les contours qui correspondent à des artefacts de découpe des poumons. 

        """
    )
    
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "gradcamViralPneumoniaBonneDetection.png"))
    st.markdown("""
                **Viral_Pneumonia**:  très hétérogène d'un échantillon à l'autre. On retrouve des activations sur les contours qui correspondent à des artefacts de découpe des poumons et sur les côtes qui correspondent à des superpositions. Par contre les détections d’opacités (blanc) au sein des parties claires ( i-e noires ) sont correctes, soit sur le bas et sur le centre de chaque poumon. Elles doivent être mal limitées et bilatérales mais pas forcément symétriques. 
            """
    )
    
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "gradcamNormalBonneDetection.png"))
    st.markdown("""
                **Normal**:  Les zones se situent aux apex ou aux bases et sur les côtes, ce qui est troublant. Les zones qui différencient devrait être les parties claires (i-e noires à la radio) 
        """)
    
    st.caption("Analyse de détections incorrectes")
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "gradcamauvaisedetection.png"))
    st.markdown("""
                **1ère colonne** : La détection du covid d’effectue sur la partie flou de la cage thoracique.. L’algorithme n’a pas capturé l’information que cette zone pouvait être floue également suite à une mauvaise mise au point ou un mouvement.

                **2nde colonne** : La détection du covid d’effectue sur des éléments non caractéristiques. Les opacités mal définies font plus penser à une opacité pulmonaire. cela fait plus penser à une infection car le volume est trop grand
        """)
    
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "radio1.png"))
    
    st.markdown("""
                    **3ème colonne** : La détection du covid d’effectue sur la partie flou de la cage thoracique. Les parties claires sont normales. Il n’y a pas d’anomalies.
    
                    **4ème colonne** : Les opacités mal définies n’ont pas été vues par l’algorithme. Il s’est focalisé sur de mauvaises parties. Ici, un spécialiste détecterait une opacité en base droite très légères, difficilement décelable car le critère serait l’aspect. 
            """)
    
    st.image(str(PROJECT_ROOT / "src" / "streamlit" / "radio2.png"))
    
if page == pages[12] :
    st.write("### Cas Pratique")

    st.markdown(
        """
Testez le modèle retenu — **CNN 4 niveaux (tuned)** — sur une radiographie de votre choix.
Le pipeline reproduit le preprocessing utilisé à l'entraînement : redimensionnement
299×299, égalisation d'histogramme, CLAHE, puis application du masque pulmonaire.
        """
    )

    default_image_path = BASE_DIR / "radio.png"
    default_mask_path = BASE_DIR / "masque.png"

    col_img, col_mask = st.columns(2)

    with col_img:
        st.write("#### Radiographie")
        fichier_image = st.file_uploader(
            "Charger une radiographie (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="cas_pratique_image",
        )
        if fichier_image is not None:
            image_source = Image.open(fichier_image)
        elif default_image_path.exists():
            st.caption(f"Aucun fichier chargé — image par défaut utilisée ({default_image_path.name}).")
            image_source = Image.open(default_image_path)
        else:
            st.warning(f"Aucune image chargée et fichier par défaut introuvable : {default_image_path}")
            image_source = None
        if image_source is not None:
            st.image(image_source, caption="Radiographie sélectionnée", width=280)

    with col_mask:
        st.write("#### Masque pulmonaire")
        fichier_masque = st.file_uploader(
            "Charger un masque (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="cas_pratique_masque",
        )
        if fichier_masque is not None:
            mask_source = Image.open(fichier_masque)
        elif default_mask_path.exists():
            st.caption(f"Aucun fichier chargé — masque par défaut utilisé ({default_mask_path.name}).")
            mask_source = Image.open(default_mask_path)
        else:
            st.warning(f"Aucun masque chargé et fichier par défaut introuvable : {default_mask_path}")
            mask_source = None
        if mask_source is not None:
            st.image(mask_source, caption="Masque sélectionné", width=280)

    def preprocess_cas_pratique(image_pil: Image.Image, mask_pil: Image.Image, taille: int = size_img):
        """Reproduit le pipeline d'entraînement : niveaux de gris, resize 299x299,
        égalisation d'histogramme, CLAHE, puis application du masque pulmonaire.
        Ajuster clipLimit/tileGridSize si différents des valeurs utilisées à l'entraînement."""
        img_resized = np.array(image_pil.convert("L").resize((taille, taille)))
        mask_resized = np.array(mask_pil.convert("L").resize((taille, taille)))

        img_eq = cv2.equalizeHist(img_resized)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_eq)

        _, mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
        img_masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask_bin)

        return img_resized, img_clahe, img_masked

    st.divider()

    classifier_disabled = image_source is None or mask_source is None
    if st.button("🔬 Classifier l'image", type="primary", disabled=classifier_disabled):
        try:
            model_cas_pratique = get_model_CNN256_tuned()
        except Exception as e:
            st.error(f"Modèle CNN 4 niveaux (tuned) indisponible : {e}")
        else:
            img_resized, img_clahe, img_masked = preprocess_cas_pratique(image_source, mask_source)

            st.write("#### Étapes du preprocessing")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_resized, caption="Redimensionnée 299×299 (niveaux de gris)", width=220)
            with col2:
                st.image(img_clahe, caption="Égalisation + CLAHE", width=220)
            with col3:
                st.image(img_masked, caption="Masque appliqué", width=220)

            entree = img_masked.astype("float32").reshape(1, size_img, size_img, 1)
            prediction = model_cas_pratique.predict(entree)
            classe_idx = int(np.argmax(prediction[0]))
            classe_predite = FROZEN_CLASS_NAMES[classe_idx]

            st.success(f"**Classe prédite : {classe_predite}**")

            proba_df = pd.DataFrame(
                {"Classe": FROZEN_CLASS_NAMES, "Probabilité": prediction[0]}
            ).sort_values("Probabilité", ascending=False)
            st.bar_chart(proba_df.set_index("Classe"))

