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

    st.write(f"Le jeu de données contient : \
        {len(normal_images)} images normales, \
        {len(covid_images)} images Covid, \
        {len(lung_images)} images Lung Opacity, \
        {len(viral_images)} images Viral pneumonia.")

    st.write("Les images COVID-19 proviennent de sources hétérogènes (PadChest, GitHub, SIRM, et autres dépôts publics), tandis que les classes Normal, Lung Opacity et Viral Pneumonia sont issues de bases de données uniques (RSNA, Kaggle).")

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

    st.write("### DataVizualisation")


if page == pages[2]:
    st.write("### Preprocessing")

    st.write("Pour répondre aux constats dressés lors de l'étape d'exploration, nous avons défini le pipeline de preprocessing ci-dessous.")

    st.image("pipeline_preprocessing.png", caption="Pipeline preprocessing")

    st.write("Cette stratégie a été adoptée pour fiabiliser le jeu de données (dé-duplication), supprimer le fond (masques), harmoniser les caractéristiques des images (normalisation) et augmenter la variabilité des données (augmentation)")

if page == pages[3]:
    st.write("### Modélisation")



    
