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
import joblib

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
model_CNN_path = "../../models/cnn256/cnn_256.keras"    
val_dir ="../../../COVID-19_Radiography_Dataset_split/validation/"  
test_dir = "../../../COVID-19_Radiography_Dataset_split/test/"
model_SVM_path = "../../models/svm/svm_weighted.joblib" 
scaler_SVM_path = "../../models/svm/scaler_svm.joblib"  
images_dir = "../../../COVID-19_Radiography_Dataset/"
csv_test="../../../features/test_features.csv"
csv_validation="../../../features/validation_features.csv"


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
st.set_page_config(page_title=" Classification de Radios pulmonaires", layout="wide")
st.title("🫁 Classification de radios pulmonaires")
st.caption("COVID / NORMAL / LUNG_OPACITY / VIRAL_PNEUMONIA")




# --------------------------------------------------------------------------- #
# Fonctions mises en cache
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle CNN …")
def get_model_CNN(path: str):
    return load_model(path)

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


# --------------------------------------------------------------------------- #
# Introduction
# --------------------------------------------------------------------------- #
if page == pages[0]:
    st.write("### Introduction")
    st.write("""L'expansion rapide de l'épidémie de COVID-19 a très vite mis les systèmes de santé sous tension. Cet épisode a montré la nécessité d'obtenir un 
        diagnostic de manière instantanée et fiable. Celui-ci repose principalement sur le technique RT-PCR (Reverse Transcription Polymerase Chain Reaction), mais des études ont aussi mis en évidence certaines limites de cette technique.
        C'est pourquoi, l'imagerie médicale est apparue comme un outil complémentaire intéressant pour détecter les cas COVID.""")
    st.write("""Notre projet propose de développer un modèle de classification automatique de radiographies pulmonaires capable de distinguer les cas COVID-19 des autres pathologies pulmonaires (pneumonie virale, opacité pulmonaire) et des poumons sains. Afin de répondre à cet objectif, nous disposions d'un jeu de données disponible ici :""")
    st.page_link("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database", label="COVID-19 Radiography Database")
    

# --------------------------------------------------------------------------- #
# DataVisualisation
# --------------------------------------------------------------------------- #
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
    
    st.write("### Distribution des pixels par classe")
    col1, col2 , col3, col4= st.columns(4)
    with col1:
        st.image("hist_intensite_1.png", caption="Sans masque")

    with col2:
        st.image("hist_intensite_2_masques.png", caption="Avec masque")

    with col3:
        st.image("hist_intensite_3.png", caption="Sans masque")

    with col4:
        st.image("hist_intensite_4_masques.png", caption="Avec masque")

    st.write("L'analyse de la luminosité des images montre des différences entre les différentes classes. La moyenne des pixels traduit le niveau global de luminosité d'une radiographie tandis que l'écart-type renseigne sur l'hétérogénité des nivveaux de gris. La classe Covid présente les intensités les plus élevées et ses valeurs sont plus dispersées tandis que les autres classes présentent des valeurs assez proches. L'apllication des masques modifient ces indicateurs en diminuant l'intensité lumineuse et en resserrant la distribution. Les différences entre les classes sont moins marquées.")


    st.write("### Analyse en composantes principales")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("ACP_1.png")

    with col2:
        st.image("ACP_3.png")

    with col3:
        st.image("ACP_4.png")

    st.write("Une analyse en composantes principales (ACP) sur le jeu de données est réalisée afin de visualiser la structure des données, repérer d'éventuels outliers et évaluer la séparation des différentes classes.")
    st.write("La forte superposition des classes suggère que les différences observées sont dues à des motifs radiologiques complexes et locaux. Cela justifie le recours a des modèles d'apprentissage capables d'extraire les caractéristiques discriminantes spécifique à chaque classe.")


    st.write("### Outliers")


    st.write("Pour chaque classe, les images dont la distance quadratique moyenne (MSE) par rapport à l'image moyenne de leur classe dépasse un seuil ont été identifiées comme atypiques :")
    st.write("- Normal : 11 outliers (MSE > 2500)")
    st.write("- COVID : 34 outliers (MSE > 4 000)")
    st.write("- Lung Opacity : 20 outliers (MSE > 4 000)")
    st.write("- Viral Pneumonia : 32 outliers (MSE > 3 000)")
    st.write("Les distances sont calculées en travaillant sur une image redimensionnée à 50x50 pixels")


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
    
    strategies_analyse=pd.DataFrame({"Stratégie":["Data augmentation", "Class weigths"],
                                        "Description":["Via ImageDataGenerator de Keras, appliquée uniquement sur le jeu d'entraînement. Elle permet d'augmenter artificiellement le nombre d'images des classes minoritaires par transformations géométriques (rotation, flip horizontal, zoom)",
                                        "Pour les modèles sensibles au déséquilibre (CNN, Random Forest), un poids inversement proportionnel à la fréquence de chaque classe sera appliqué dans la fonction de perte, forçant le modèle à accorder plus d'importance aux classes rares comme Viral Pneumonia."]
                                        })
    
    deduplication=pd.DataFrame({"Classe":["Covid", "Normal", "Lung Opacity", "Viral Pneumonia"],
                                    "Doublons":[1, 223, 0, 7],
                                    "outliers":[23, 55, 24, 0]})
    
    st.write("### Preprocessing")
    
    st.write("Pour répondre aux constats dressés lors de l'étape d'exploration, nous avons défini le pipeline de preprocessing ci-dessous.")
    
    st.image("pipeline_preprocessing.png", caption="Pipeline preprocessing")
    
    st.write("Cette étape comprend les processus de nettoyage, de traitement des images, de séparation et enfin d'augmantation sur le jeu d'entraînement uniquement.")
    
    st.write("Cette stratégie a été adoptée pour fiabiliser le jeu de données (dé-duplication), supprimer le fond (masques), harmoniser les caractéristiques des images (normalisation) et augmenter la variabilité des données (augmentation)")
    
    st.write("L'étape d'exploration a permis de repérer 231 doublons et 102 outliers")
    st.table(deduplication)
    
    st.write("### Génération d'un dataset des caractéristiques des images")
    
    st.write("Un datase tabulaire a été généré en vue de l'entraînement des modèles de machine learning. Il contient l'ensemble des features des images traitées :")
    st.table(features)
    
    st.write("Le jeu de données ainsi constitué compte 20 835 images réparties sur les 4 classes.")
    
    st.write("### Gestion des déséquilibres")
    
    st.write("Deux stratégies ont été définies pour prendre en compte le déséquilibre des classes constaté lors de l'étape d'analyse :")
    st.table(strategies_analyse)
    

# --------------------------------------------------------------------------- #
# Modèle SVM
# --------------------------------------------------------------------------- #
if page == pages[3] : 
    st.write("### Modèle SVM")

    try:
        model_loaded = get_model_SVM(model_SVM_path)
    except Exception as e:
        st.error(f"Impossible de charger le modèle SVM: {e}")
        st.stop()

    try:
        scaler_loaded = get_scaler_SVM(scaler_SVM_path)
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

    with st.spinner("Prédiction sur le jeu de test…"):
        test_pred_class = model_loaded.predict(X_test_scaled)

    with st.spinner("Prédiction sur le jeu de validation…"):
        val_pred_class = model_loaded.predict(X_val_scaled)

    class_names = sorted(y_test.unique())

    tab_val, tab_test = st.tabs(["Validation", "Test"])
    with tab_val:
        evaluer(y_val, val_pred_class, class_names, "Résultats — Validation")
    with tab_test:
        evaluer(y_test, test_pred_class, class_names, "Résultats — Test")


# --------------------------------------------------------------------------- #
# Modèle CNN
# --------------------------------------------------------------------------- #
if page == pages[4] : 
    st.write("### Modèle CNN 4 niveaux")
    try:
        model_loaded = get_model_CNN(model_CNN_path)
    except Exception as e:
        st.error(f"Impossible de charger le modèle  CNN256 : {e}")
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

