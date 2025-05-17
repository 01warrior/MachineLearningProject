# Projet d'Analyse de la Dépression Étudiante

## Description
Ce projet utilise le machine learning pour analyser et prédire la dépression chez les étudiants en se basant sur divers facteurs comme la pression académique, le sommeil et le mode de vie. Le modèle atteint une précision de 85% et un score ROC AUC de 92%.

## Structure du Projet
- `ProjetClasse.ipynb`: Notebook principal contenant l'analyse complète
- `Student Depression Dataset.csv`: Données source
- `/fichierEnregistrement`: Dossier contenant les fichiers du modèle
  - Modèle entraîné (.joblib)
  - Fichiers de prétraitement
  - Application Streamlit

## Caractéristiques Principales
- Nettoyage et prétraitement des données
- Analyse de corrélation approfondie
- Encodage des variables catégorielles
- Modélisation par régression logistique
- Performance :
  - Accuracy: 85.03%
  - ROC AUC Score: 92.32%

## Technologies Utilisées
- Python 3.x
- Pandas et NumPy
- Scikit-learn
- Seaborn et Matplotlib
- Streamlit
- Joblib

## Modèle
Le modèle de régression logistique a été choisi pour sa simplicité et son efficacité. Les résultats montrent une excellente capacité à distinguer les cas de dépression.

## Installation et Utilisation
1. Cloner le repository
2. Installer les dépendances
3. Exécuter le notebook ou l'application Streamlit

Projet réalisé dans le cadre du cours de Machine Learning à Mundiapolis.
