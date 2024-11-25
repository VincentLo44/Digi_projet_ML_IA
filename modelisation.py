import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def modelisation(data, data_target_column):
    st.title("Analyse de données avec Streamlit")
    # Gestion des valeurs manquantes
    st.header("Gestion des valeurs manquantes")
    st.write("Nombre de valeurs manquantes par colonne :")
    st.write(data_target_column.isnull().sum())

    # Option pour imputer les colonnes numériques avec la moyenne
    if st.checkbox("Imputer les valeurs manquantes avec la moyenne pour les colonnes numériques"):
        # Identifier les colonnes numériques
        numeric_columns = data_target_column.select_dtypes(include=["float64", "int64"]).columns
        # Remplir uniquement les colonnes numériques avec la moyenne
        data_target_column[numeric_columns] = data_target_column[numeric_columns].fillna(data_target_column[numeric_columns].mean())
        st.write("Données après imputation des colonnes numériques :")
        st.write(data_target_column.head())

# Vérifiez que la colonne 'alcohol' et 'target' existent
    if 'alcohol' in data.columns and 'target' in data.columns:
        # Interaction utilisateur : Définir la plage de degré d'alcool
        min_alcohol = int(np.floor(data['alcohol'].min()))
        max_alcohol = int(np.ceil(data['alcohol'].max()))
        
        # Ajout d'une entrée interactive pour ajuster la plage
        # Ordonnées
        min_input, max_input = st.slider(
            "Sélectionnez la plage de degré d'alcool",
            min_value=min_alcohol,
            max_value=max_alcohol,
            value=(min_alcohol, max_alcohol)
        )

        # Définir le nombre de bins (peut être ajusté dynamiquement)
        # Abscisses
        bins = st.slider(
            "Nombre de bins pour l'histogramme",
            min_value=5,
            max_value=50,
            value=10
        )
        
        # Création du graphique
        fig, axs = plt.subplots(1, 3, figsize=(20, 10), sharey=True)
        fig.suptitle("Distribution du degré d'alcool pour les différents types de vins", fontsize=13)

        # Boucle sur chaque type de vin pour créer les histogrammes
        targets = pd.unique(data['target'])
        for cpt, target in enumerate(targets):
            if cpt < 3:  # Limité à 3 sous-graphiques pour correspondre à axs
                temp = data[data['target'] == target]
                hist, bins_edges = np.histogram(temp['alcohol'], bins=bins, range=[min_input, max_input])
                
                label = f"Type : {str(target)}"
                axs[cpt].hist(temp['alcohol'], bins=bins_edges, histtype='stepfilled', label=label, color='skyblue', edgecolor='black')
                axs[cpt].set_xlabel("Degré d'alcool")
                axs[cpt].set_ylabel("Nombre de vins")
                axs[cpt].legend()

        # Alignement des labels
        fig.align_ylabels(axs)
        
        # Afficher dans Streamlit
        st.pyplot(fig)
    else:
        st.write("Les colonnes 'alcohol' et/ou 'target' sont manquantes dans le DataFrame.")


    # Option pour afficher les colonnes non numériques avec des valeurs manquantes
    if st.checkbox("Afficher les colonnes non numériques avec des valeurs manquantes"):
        non_numeric_columns = data_target_column.select_dtypes(exclude=["float64", "int64"]).columns
        missing_non_numeric = data_target_column[non_numeric_columns].isnull().sum()
        st.write("Colonnes non numériques avec des valeurs manquantes :")
        st.write(missing_non_numeric)

        # Option pour imputer les colonnes non numériques avec une valeur par défaut
        if st.checkbox("Imputer les colonnes non numériques avec une valeur par défaut (ex: 'inconnu')"):
            data[non_numeric_columns] = data[non_numeric_columns].fillna("inconnu")
            st.write("Données après imputation des colonnes non numériques :")
            st.write(data_target_column.head())

    # Visualisation des statistiques de base
    st.header("Statistiques de base")
    st.write(data_target_column.describe())

    # Option pour afficher les corrélations entre les colonnes numériques
    if st.checkbox("Afficher les corrélations entre les colonnes numériques"):
        # Vérification des colonnes numériques uniquement
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        # Suppression de la première colonne (assumée comme étant l'index ou à exclure)
        numeric_data = numeric_data.iloc[:,1:]  # Exclut la première colonne
        
        if numeric_data.empty:
            st.write("Aucune colonne numérique disponible pour calculer les corrélations.")
        else:
            st.write("Corrélations :")
            corr = numeric_data.corr()
            st.write(corr)

            fig, ax = plt.subplots(figsize=(10, 8))  # Taille du graphique plus grande
            sns.set(style="whitegrid")  # Application du thème Seaborn
            mat_corr = sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", ax=ax, 
                                cbar_kws={"shrink": 0.8}, linewidths=0.5)  # Amélioration du style du heatmap
            ax.set_title("Matrice de Corrélation", fontsize=16, weight='bold')  # Ajout du titre
            st.pyplot(fig)


    # Visualisation des relations entre les variables numériques (Pairplot)
    if st.checkbox("Afficher le pairplot des variables numériques"):
        # Vérification de la présence de colonnes numériques avant d'afficher le pairplot
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        if not numeric_data.empty:
            st.write("Visualisation des relations entre les variables numériques :")
            pairplot_fig = sns.pairplot(numeric_data)
            st.pyplot(pairplot_fig)
        else:
            st.write("Aucune colonne numérique disponible pour afficher le pairplot.")

    # Option pour sauvegarder les données transformées
    if st.checkbox("Sauvegarder les données transformées"):
        csv = data.to_csv(index=False)
        st.download_button(
            label="Télécharger les données transformées",
            data=csv,
            file_name="data_transformed.csv",
            mime="text/csv",
        )
    else:
        st.write("Veuillez charger un fichier CSV pour commencer.")
