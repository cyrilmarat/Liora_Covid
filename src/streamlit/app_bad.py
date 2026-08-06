"""
Application Streamlit — Évaluation d'un modèle CNN (radiographies pulmonaires)
Reprend le pipeline du notebook plot_model_cnn_256.ipynb :
  1. Chargement des jeux de données (validation / test) depuis des dossiers d'images
  2. Chargement du modèle .keras entraîné (avec la métrique custom SparseF1Score)
  3. Prédiction sur validation et test
  4. Affichage : accuracy, classification_report, matrice de confusion

Lancement :
    streamlit run app.py
"""

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


# --------------------------------------------------------------------------- #
# Métrique custom nécessaire pour désérialiser le modèle (identique au notebook)
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
# Sidebar : paramètres
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Paramètres")

    model_path = st.text_input(
        "Chemin du modèle (.keras)",
        value="cnn_256.keras",
        help="Chemin vers le fichier modèle sauvegardé (ex: cnn_256.keras)",
    )

    val_dir = st.text_input(
        "Dossier validation",
        value="../../../COVID-19_Radiography_Dataset_split/validation/",
    )
    test_dir = st.text_input(
        "Dossier test",
        value="../../../COVID-19_Radiography_Dataset_split/test/",
    )

    col1, col2 = st.columns(2)
    with col1:
        img_h = st.number_input("Hauteur image", value=299, step=1)
    with col2:
        img_w = st.number_input("Largeur image", value=299, step=1)

    batch_size = st.number_input("Batch size", value=32, step=1)
    color_mode = st.selectbox("Color mode", ["grayscale", "rgb"], index=0)

    save_npy = st.checkbox(
        "Sauvegarder y_true / y_prob (test) au format .npy", value=False
    )

    run_btn = st.button("🚀 Charger le modèle et évaluer", type="primary")


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
if run_btn:
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
        val_ds = get_dataset(val_dir, img_h, img_w, batch_size, color_mode)
        test_ds = get_dataset(test_dir, img_h, img_w, batch_size, color_mode)
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

    if save_npy:
        np.save("y_true_test.npy", y_true_test_class)
        np.save("y_prob_test.npy", test_pred)
        st.success("Fichiers y_true_test.npy et y_prob_test.npy sauvegardés.")

    tab_val, tab_test = st.tabs(["Validation", "Test"])
    with tab_val:
        evaluer(y_true_val_class, val_pred_class, class_names, "Résultats — Validation")
    with tab_test:
        evaluer(y_true_test_class, test_pred_class, class_names, "Résultats — Test")

else:
    st.info("Renseigne les chemins dans la barre latérale puis clique sur **Charger le modèle et évaluer**.")
