# Contexte et analyse du dataset - Maxime Vigier

## Dataset : COVID-19 Radiography Database
Créé par Qatar University + University of Dhaka + collaborateurs Pakistan/Malaisie.

## Composition du dataset
- COVID : 3 616 images
- Normal : 10 192 images
- Lung Opacity : 6 012 images
- Viral Pneumonia : 1 345 images
- **Total : 21 165 images**

## Problèmes identifiés

### 1. Déséquilibre de classes
Les classes ne sont pas équilibrées — Normal est très majoritaire (10 192) 
et Viral Pneumonia très minoritaire (1 345).
→ À gérer en pre-processing (data augmentation, class weights)

### 2. Sources hétérogènes
Les images Covid viennent de 4 sources différentes 
(PadChest, Github, SIRM, Kaggle) → qualité et style d'images variables.
→ Risque que le modèle apprenne des artefacts techniques 
plutôt que des caractéristiques médicales.

### 3. Dimensions masques ≠ images
- Images : 299×299 pixels
- Masques : 256×256 pixels
→ Redimensionnement nécessaire avant d'appliquer les masques
→ Solution : cv2.resize + cv2.bitwise_and (voir notebook Cyril)

### 4. Images complexes
Certaines radiographies contiennent des fils de monitoring, 
dispositifs médicaux ou annotations de radiologues 
→ Peut perturber l'apprentissage du modèle

## Bibliographie

### Articles de création du dataset
- Chowdhury et al. "Can AI help in screening Viral and COVID-19 pneumonia?" 
  IEEE Access, Vol. 8, 2020, pp. 132665-132676
- Rahman et al. "Exploring the Effect of Image Enhancement Techniques 
  on COVID-19 Detection using Chest X-ray Images." arXiv:2012.02238, 2020

### Articles méthodologiques
- COVID-CT Dataset : https://arxiv.org/abs/2003.13865
- Effect of image enhancement : https://doi.org/10.1016/j.compbiomed.2021.105002
