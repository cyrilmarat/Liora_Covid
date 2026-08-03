import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
import joblib

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
model_CNN_path = "../../models/cnn256/cnn_256.keras"    
val_dir ="../../../COVID-19_Radiography_Dataset_split/validation/"  
test_dir = "../../../COVID-19_Radiography_Dataset_split/test/"
model_SVM_path = "../../models/svm/svm_weighted.joblib"
scaler_SVM_path = "../../models/svm/scaler_svm.joblib"
csv_test="../../../features/test_features.csv"
csv_validation="../../../features/validation_features.csv"

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
st.set_page_config(page_title=" Classification de Radios pulmonaires", layout="wide")
st.title("🫁 Classification de radiographies pulmonaires")
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
pages=["1.Introduction", "2.Données & Visualisation", "3.Preprocessing", "4.Vers la modélisation", "5.Modèles & résultats", "6.ML: Modèle SVM", "7.DL: Modèle CNN 4 niveaux", "8.Biais de source", "9.Analyse du meilleur modèle", "10.Conclusion", "11.Limites & perspectives"]
page=st.sidebar.radio("Aller vers", pages)


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

    st.image("distribution.png", caption="Répartition des classes")

    st.write("Les images COVID-19 proviennent de sources hétérogènes (PadChest, GitHub, SIRM, et autres dépôts publics), tandis que les classes Normal, Lung Opacity et Viral Pneumonia sont issues de bases de données uniques (RSNA, Kaggle).")

    st.write("#### Exemples d'images du dataset et les masques associés")
    st.image("../../reports/figures/exemples_images_masques.png")

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
    st.image("../../reports/figures/exemple_pipeline.png", caption="Exemple de preprocessing sur une image COVID", width=650)

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
if page == pages[4] :
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
    st.image("../../reports/figures/rpt_f1_global.png", caption="F1-macro — comparaison Machine Learning vs Deep Learning (test)")

    st.write("#### Tableau récapitulatif final")
    recap = pd.read_csv("../../models/model_comparison_recap_final.csv")
    st.dataframe(recap, hide_index=True, use_container_width=True, height=420)


# --------------------------------------------------------------------------- #
# Modèle SVM
# --------------------------------------------------------------------------- #
if page == pages[5] :
    st.write("### ML Modèle SVM")

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
if page == pages[6] :
    st.write("### DL Modèle CNN 4 niveaux")
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
    st.image("../../reports/figures/source_bias_covid_recall.png", width=750)
    st.caption(
        "Rappel COVID très variable selon la source pour les modèles Machine Learning "
        "(à nuancer : Eurorad n=30 et SIRM n=14 sont de petits effectifs)."
    )


# --------------------------------------------------------------------------- #
# Analyse du meilleur modèle
# --------------------------------------------------------------------------- #
if page == pages[8] :
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
    col1, col2 = st.columns(2)
    with col1:
        st.image("../../reports/figures/cm_deep_learning_normalized.png", caption="Matrices de confusion normalisées — Deep Learning")
    with col2:
        st.image("../../reports/figures/rpt_pr_curves_dl.png", caption="Courbes Précision-Rappel — Deep Learning")

    st.write("#### Grad-CAM — où le modèle « regarde »")
    st.image("../../reports/figures/gradcam_planche.png", caption="Cartes d'activation Grad-CAM par modèle", width=750)


# --------------------------------------------------------------------------- #
# Conclusion
# --------------------------------------------------------------------------- #
if page == pages[9] :
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
if page == pages[10] :
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
- **Validation externe** : le modèle retenu n'a pas été testé sur des données d'une autre
  origine que ce dataset.
        """
    )

