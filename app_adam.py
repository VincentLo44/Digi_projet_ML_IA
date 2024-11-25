import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns


# Afficher un graphique
#fig, ax = plt.subplots()
#ax.plot(data["Année"], data["Ventes"], marker="o")
#ax.set_title("Ventes par année")
#ax.set_xlabel("Année")
#ax.set_ylabel("Ventes")
#st.pyplot(fig)



st.set_page_config(
    page_title="Projet ML",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)
    
# Import dataset
df = pd.read_csv("./Data/vin.csv")

tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Traitement des données", "Visualisations", "Modelisation", "Evaluation"])

#######################################################
# 2/ Chargement du jeu de donnée : 
# Charger vos jeux csv à partir de chemins locaux dans votre application afin de construire vos dataframe. 
# A partir de là, vous devez intégrer des interactions utilisateurs
#######################################################
with tabs_1:
    
    # Afficher le tableau
    st.title("Chargement du jeu de donnée")
    st.header("Affichage du tableau")
    st.write("Voici un aperçu du dataframe directement après son importation :")
    st.dataframe(df.head())
    st.write("Nous pouvons constater que la première colonne n'est pas été définie en tant qu'index.")
    
    # Configurer l'index
    st.header("Définition de l'index")
    st.markdown("Nous pouvons le faire en utilisant la commande `.set_index()`.")
    df.columns = ["index","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium","total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","od280/od315_of_diluted_wines","proline","target"]
    st.write("Voici un aperçu du dataframe directement après ce petit traitement :")
    df = df.set_index("index")

    # Afficher le nouveau tableau
    st.dataframe(df.head())
    
    # STATS DESCRIPTIVES
    st.title("Analyse descriptive du dataframe")
    
    st.header("Types des colonnes")
    st.write(df.dtypes.to_frame().T)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Variables catégorielles:")
        st.write(df["target"].value_counts().to_frame().T)

        st.subheader("Valeurs de 'target' manquantes")
        st.write(df["target"].isna().sum())
    
    with col2:
        st.header("Variables quantitatives:")
        # df uniquement les colonnes numériques
        df_quant = df.drop("target",axis=1)
        variable_quantitative = st.selectbox("Sélectionnez une variable :", df_quant.columns)

        st.write(df_quant[variable_quantitative].describe(include='all').to_frame().T)

        st.subheader(f"Valeurs de '{variable_quantitative}' manquantes")
        st.write(df_quant[variable_quantitative].isna().sum())
        
    st.subheader("Matrice de correlation")
    corr = df_quant.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan
    st.dataframe(
        corr.style
        .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
        .highlight_null(color='#f1f1f1')
        .format(precision=2)
        )

#######################################################
# 3/ Effectuer un bloc de traitement de donnée : essayez d’intégrer le maximum de fonctionnalité
# dans le traitement des données, voici quelques idées intéressantes à implémenter :
# - Analyse descriptive du dataframe
# - Graphique de distribution et pairplot
# - Corrélation avec la target
# - Fréquences
# - Standardisation
# - Valeur manquante et gestion
# Il serait intéressant d’intégrer des interaction utilisateurs, comme par exemple permettre à
# l’utilisateur de sélectionner les colonnes souhaitées et appliqué sa demande (afficher un
# graphique, supprimer une colonne, imputer les valeurs manquantes par un choix etc)
#######################################################
with tabs_2:    
    st.title("Variables qualitatives")
    st.header("target")
    st.bar_chart(df["target"].value_counts())

    st.title("Variables quantitatives")
    colonne_selection = st.selectbox("Sélectionnez une variable :", df_quant.columns, key="col_select")
    graphe_selection = st.selectbox("Sélectionnez un graphe :", ["Histogramme","Boxplot","Courbe de densité","Nuage de point"], 
                                    key="graph_select")
    
    st.header(f"Variable **{colonne_selection}**")
    
    if graphe_selection == "Histogramme":
        fig, ax = plt.subplots()
        bins = st.slider('Number of bins', min_value=5, max_value=len(df), value=20, step=1)
        ax.hist(df[colonne_selection], bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution de la colonne {colonne_selection}")
        ax.set_xlabel(colonne_selection)
        ax.set_ylabel('Fréquence')
        st.pyplot(fig)
        
    elif graphe_selection == "Boxplot":     
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='target', y=colonne_selection, palette="Set2", ax=ax)
        ax.set_title(f"Boxplot de {colonne_selection} selon le Target")
        ax.set_xlabel("Target")
        ax.set_ylabel(colonne_selection)
        st.pyplot(fig)
        
    elif graphe_selection == "Courbe de densité": 
        fig, ax = plt.subplots()
        sns.kdeplot(df[colonne_selection], fill=True, color="skyblue", ax=ax)
        ax.set_title(f"Densité de la colonne {colonne_selection}")
        ax.set_xlabel(colonne_selection)
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    
    elif graphe_selection == "Nuage de point":    
        st.scatter_chart(df, x="target", y=colonne_selection)

with tabs_3:

    #######################################################
    # 4/ Effectuer un bloc machine learning pipeline : essayer d’intégrer un pipeline de traitement avec
    # la possibilité de laisser choisir entre plusieurs algorithmes selon la target détecté. Ensuite,
    # appliquer le split, le fit et les prédictions des données. Vous pouvez permettre à l’utilisateur de
    # prédire sur de nouvelles données, d’enregistrer le model etc
    #######################################################
    pass

with tabs_4:
    
    #######################################################
    # 5/ Effectuer un bloc d’évaluation : essayer d’intégrer un bloc d’évaluation du model qui vient de
    # tourner et de s’entrainer. Vous pouvez utiliser les métrics ou bien des graphiques. 
    #######################################################
    pass


    
#######################################################
# 6/ BONUS : Ajouter des fonctionnalités supplémentaires : essayer d’ajouter des fonctionnalités
# pour optimiser l’application comme par exemple, un lazy predict, un gridSearchCV, l’integration
# d’un modèle de Deep Learning etc. pas de limite …
#######################################################