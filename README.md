# Diginamic Projet Machine Learning IA
Projet Diginamic sur le développement d'IA avec Streamlit et python

## Notes sur le processus
Le processus d'élaboration de ce projet d'IA consistait à développer un site web interactive Python qui utilise la librairie Streamlit.
Le script Python génère une page web qui nous génère des analyses de données et des modèles de machine learning à partir d'un fichier qu'on fournit à l'application.

Le projet se compose de 6 scripts Python : 
- **app.py** : point d'entrée qui centralise et orchestre les différentes fonctions en provenance des autres scripts **.py**
- **upload_data.py** : script pour importer des données sous forme de .csv
- **modelisation.py** : affichage des données + gestion des valeurs manquantes + statistiques de base
- **general_analysis.py** : choix de variable à analyser + choix de graphe à afficher + matrice de corrélation et pairplot
- **machine_learning.py** : choix de variable à expliquer + choix de modèle de régression + téléchargement du modèle généré
- **preexistant.py** : chargement d'un modèle pré-existant + test du modèle en choisissant les valeurs des variables explicatives

## L’équipes
L'équipe du projet se compose de : 
- BOUTE Benjamin
- LOREAU Vincent
- MIGLIACCIO Melody
- OURRAD Adam

## La répartition du travail
Prévisualisation des données : Adam & Benjamin
Analyse généralisée : Adam & Benjamin 
Machine Learning : Melody & Vincent
Modèle préexistant : Melody & Vincent 

## Mode d’emploi pour utiliser votre application
L'application a été déployé vers le lien suivant : https://xkc7vpmzqitowvkchesrdm.streamlit.app/

Une fois sur la page web, sur la sidebar à gauche, il faut y charger un fichier **.csv** de notre choix.

Il faut naviguer dans les 4 onglets suivants : 

- **Prévisualisation des données** : 
    + affichage des données 
    + gestion des valeurs manquantes 
    + statistiques de base

- **Analyse généralisée** : 
    + choix de variable à analyser 
    + choix de graphe à afficher 
    + matrice de corrélation et pairplot

- **Machine Learning** : 
    + choix de variable à expliquer 
    + choix de modèle de régression 
    + téléchargement du modèle généré

- **Modèle préexistant** : 
    + chargement d'un modèle pré-existant 
    + test du modèle en choisissant les valeurs des variables explicatives