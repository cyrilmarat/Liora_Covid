import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
import cv2 #import OpenCV


# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
model_path = "../../models/cnn256/cnn_256.keras"    
val_dir ="../../../COVID-19_Radiography_Dataset_split/validation/"  
test_dir = "../../../COVID-19_Radiography_Dataset_split/test/"
    






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
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[2] : 
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
        val_ds = get_dataset(val_dir, img_h=299,img_w=299, batch_size=32, color_mode='grayscale')
        test_ds = get_dataset(test_dir,img_h=299,img_w=299,  batch_size=32, color_mode='grayscale')
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

