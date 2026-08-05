"""Résultats figés pour les modèles Régression Logistique et XGBoost.

Contrairement au SVM et au CNN (pages qui rechargent le modèle et relancent une
prédiction à chaque affichage), ces deux modèles n'ont pas d'artefact entraîné
et sauvegardé de manière fiable au moment de la préparation de la soutenance.
On fige donc ici les matrices de confusion telles que rapportées dans
`models/model_comparison_recap_final.csv` (issues des sorties enregistrées de
`models/regression_logistique_baseline.ipynb` et `models/xgboost_baseline.ipynb`),
afin que les résultats affichés en soutenance restent identiques au tableau
récapitulatif, quelle que soit la version de scikit-learn/xgboost installée au
moment de la présentation.

Chaque matrice est un tableau 4x4 [ligne = classe réelle, colonne = classe prédite],
dans l'ordre CLASS_NAMES (ordre alphabétique, cohérent avec le reste de l'app).
"""

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

LOGREG_RESULTS = {
    "Sans pondération": {
        "Test": [
            [0, 131, 204, 2],
            [0, 346, 242, 11],
            [0, 123, 886, 5],
            [0, 38, 78, 18],
        ],
    },
    "Balanced (avec pondération)": {
        "Test": [
            [46, 138, 120, 33],
            [47, 350, 117, 85],
            [108, 164, 590, 152],
            [2, 15, 21, 96],
        ],
    },
}

XGBOOST_RESULTS = {
    "Baseline": {
        "Validation": [
            [26, 117, 184, 11],
            [39, 308, 228, 23],
            [29, 95, 869, 20],
            [6, 27, 57, 44],
        ],
        "Test": [
            [39, 117, 173, 8],
            [28, 317, 236, 18],
            [28, 123, 839, 24],
            [9, 22, 52, 51],
        ],
    },
    "GridSearch balanced": {
        "Validation": [
            [77, 115, 115, 31],
            [94, 296, 130, 78],
            [146, 92, 643, 132],
            [8, 11, 23, 92],
        ],
        "Test": [
            [90, 102, 105, 40],
            [95, 294, 134, 76],
            [122, 110, 641, 141],
            [10, 9, 15, 100],
        ],
    },
}
