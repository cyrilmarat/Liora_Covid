import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
import cv2 #import OpenCV
import random

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
model_path = "../../models/cnn256/cnn_256.keras"    
val_dir ="../../../COVID-19_Radiography_Dataset_split/validation/"  
test_dir = "../../../COVID-19_Radiography_Dataset_split/test/"
    
images_dir = "../../../COVID-19_Radiography_Dataset/"


covid_images = list(Path(images_dir, "COVID/images").glob("*.png"))
normal_images = list(Path(images_dir, "Normal/images").glob("*.png"))
lung_images = list(Path(images_dir, "Lung_Opacity/images").glob("*.png"))
viral_images = list(Path(images_dir, "Viral_Pneumonia/images").glob("*.png"))

# --------------------------------------------------------------------------- #
# Definitions globales
# --------------------------------------------------------------------------- #
classes = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
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
st.set_page_config(page_title="Évaluation CNN - Radios pulmonaires", layout="wide")
st.title("🫁 Évaluation d'un modèle CNN — Classification de radios pulmonaires")
st.caption("COVID / NORMAL / LUNG_OPACITY / VIRAL_PNEUMONIA")




# --------------------------------------------------------------------------- #
# Fonctions mises en cache
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model(path: str):
    return load_model(path)


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


def evaluer(y_true, y_pred, class_names, titre):
    """Affiche accuracy, classification_report et matrice de confusion dans Streamlit."""
    acc = accuracy_score(y_true, y_pred)

    st.subheader(titre)
    st.metric("Accuracy", f"{acc:.4f}")

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    st.dataframe(pd.DataFrame(report_dict).transpose().round(3))

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 5))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cm_display.plot(ax=ax, colorbar=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    return acc


# --------------------------------------------------------------------------- #
# Exécution
# --------------------------------------------------------------------------- #

st.sidebar.title("Sommaire")
pages=["Introduction", "DataVisualisation", "Feature Extraction", "Modèle SVM", "Modèle CNN 4 niveaux"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write("### Introduction")
    st.write("""L'expansion rapide de l'épidémie de COVID-19 a très vite mis les systèmes de santé sous tension. Cet épisode a montré la nécessité d'obtenir un 
        diagnostic de manière instantanée et fiable. Celui-ci repose principalement sur le technique RT-PCR (Reverse Transcription Polymerase Chain Reaction), mais des études ont aussi mis en évidence certaines limites de cette technique.
        C'est pourquoi, l'imagerie médicale est apparue comme un outil complémentaire intéressant pour détecter les cas COVID.""")
    st.write("""Notre projet propose de développer un modèle de classification automatique de radiographies pulmonaires capable de distinguer les cas COVID-19 des autres pathologies pulmonaires (pneumonie virale, opacité pulmonaire) et des poumons sains. Afin de répondre à cet objectif, nous disposions d'un jeu de données disponible ici :""")
    st.page_link("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database", label="COVID-19 Radiography Database")
    


if page == pages[1] : 
    
    st.write("### Premier niveau d'analyse")

    st.write(f"Le jeu de données dont nous disposons contient 20 835 images réparties en 4 classes : Covid, Lung Opacity, Normal, Viral Pneumonia")

    st.write(f"Le jeu de données contient : \
        {len(normal_images)} images normales, \
        {len(covid_images)} images Covid, \
        {len(lung_images)} images Lung Opacity, \
        {len(viral_images)} images Viral pneumonia.")

    st.write("Les images COVID-19 proviennent de sources hétérogènes (PadChest, GitHub, SIRM, et autres dépôts publics), tandis que les classes Normal, Lung Opacity et Viral Pneumonia sont issues de bases de données uniques (RSNA, Kaggle).")

    st.write("#### Exemples d'images du dataset et les masques associés")
    fig=plt.figure (figsize=(8,16))

    for i, classe in enumerate(classes):
        dossier_images=Path(images_dir)/classe/"images"
        dossier_masques=Path(images_dir)/classe/"masks"

        image_path = random.choice(list(dossier_images.glob("*.png")))
        mask_path = dossier_masques / image_path.name

        image=cv2.imread(image_path)
        masque=cv2.imread(mask_path)

        plt.subplot(4,2,2*i+1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Classe : {classe}")

        plt.subplot(4,2,2*i+2)
        plt.imshow(masque)
        plt.axis("off")
        plt.title(f"Classe : {classe}")

    st.pyplot(fig)

if page == pages[2] : 
    st.write("### Feature Extraction")


if page == pages[3] : 
    st.write("### Modèle SVM")
    


if page == pages[4] : 
    st.write("### Modèle CNN 4 niveaux")
    try:
        model_loaded = get_model(model_path)
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

    with st.expander("📋 Résumé du modèle"):
        summary_lines = []
        model_loaded.summary(print_fn=lambda x: summary_lines.append(x))
        st.code("\n".join(summary_lines))

    try:
        val_ds = get_dataset(val_dir, img_h=size_img,img_w=size_img, batch_size=32, color_mode='grayscale')
        test_ds = get_dataset(test_dir,img_h=size_img,img_w=size_img,  batch_size=32, color_mode='grayscale')
    except Exception as e:
        st.error(f"Impossible de charger les jeux de données : {e}")
        st.stop()

    class_names = test_ds.class_names

    with st.spinner("Prédiction sur le jeu de test…"):
        test_pred = model_loaded.predict(test_ds)
        test_pred_class = test_pred.argmax(axis=1)
        y_true_test_class = np.concatenate([labels for _, labels in test_ds], axis=0)

    with st.spinner("Prédiction sur le jeu de validation…"):
        val_pred = model_loaded.predict(val_ds)
        val_pred_class = val_pred.argmax(axis=1)
        y_true_val_class = np.concatenate([labels for _, labels in val_ds], axis=0)


    tab_val, tab_test = st.tabs(["Validation", "Test"])
    with tab_val:
        evaluer(y_true_val_class, val_pred_class, class_names, "Résultats — Validation")
    with tab_test:
        evaluer(y_true_test_class, test_pred_class, class_names, "Résultats — Test")

