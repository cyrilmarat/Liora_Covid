import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import cv2
import random

from keras.utils import image_dataset_from_directory

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


chemin="/Users/antoine/Desktop/Liora/Projet/COVID-19_Radiography_Dataset"

size_img=299

st.title("Analyse des radiographies pulmonaires Covid-19")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration", "Preprocessing", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)


#Chargement du jeu de données
dataset = image_dataset_from_directory(
    directory=chemin,                                 
    image_size=(224, 224),
    batch_size = 32,
    labels="inferred",
    shuffle=True,
   color_mode='rgb' 
)


#Variables de la partie exploration
classes=dataset.class_names

covid_images = list(Path(chemin, "COVID/images").glob("*.png"))
normal_images = list(Path(chemin, "Normal/images").glob("*.png"))
lung_images = list(Path(chemin, "Lung_Opacity/images").glob("*.png"))
viral_images = list(Path(chemin, "Viral_Pneumonia/images").glob("*.png"))




if page == pages[0]:
    st.write("### Introduction")
    st.write("""L'expansion rapide de l'épidémie de COVID-19 a très vite mis les systèmes de santé sous tension. Cet épisode a montré la nécessité d'obtenir un 
        diagnostic de manière instantanée et fiable. Celui-ci repose principalement sur le technique RT-PCR (Reverse Transcription Polymerase Chain Reaction), mais des études ont aussi mis en évidence certaines limites de cette technique.
        C'est pourquoi, l'imagerie médicale est apparue comme un outil complémentaire intéressant pour détecter les cas COVID.""")
    st.write("""Notre projet propose de développer un modèle de classification automatique de radiographies pulmonaires capable de distinguer les cas COVID-19 des autres pathologies pulmonaires (pneumonie virale, opacité pulmonaire) et des poumons sains. Afin de répondre à cet objectif, nous disposions d'un jeu de données disponible ici :""")
    st.page_link("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database", label="COVID-19 Radiography Database")
    

if page == pages[1]:
    st.write("### Premier niveau d'analyse")

    st.write(f"Le jeu de données dont nous disposons contient 20 835 images réparties en 4 classes : Covid, Lung Opacity, Normal, Viral Pneumonia")


    st.image("distribution.png", caption="Répartition des différentes classes")
    st.write(f"On remarque ici une distribution déséquilibrée des classes puisque le jeu de données contient :")
    st.write(f"- {len(normal_images)} images normales ainsi que leurs masques associés")
    st.write(f"- {len(covid_images)} images Covid ainsi que leurs masques associés")
    st.write(f"- {len(lung_images)} images Lung Opacity ainsi que leurs masques associés")
    st.write(f"- {len(viral_images)} images Viral pneumonia ainsi que leurs masques associés")

    st.write("Les images COVID-19 proviennent de sources hétérogènes (PadChest, GitHub, SIRM, et autres dépôts publics), tandis que les classes Normal, Lung Opacity et Viral Pneumonia sont issues de bases de données uniques (RSNA, Kaggle). Les images sont en niveau de gris, au format PNG et ont une résolution de 299x299. La dimension des masques est plus petite (256x256).")

    st.write("#### Exemples d'images du dataset et les masques associés")
    fig=plt.figure (figsize=(8,16))

    for i, classe in enumerate(classes):
        dossier_images=Path(chemin)/classe/"images"
        dossier_masques=Path(chemin)/classe/"masks"

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


if page == pages[2]:
    
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

if page == pages[3]:
    st.write("### Modélisation")



    
