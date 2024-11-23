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
df = pd.read_csv("vin.csv")

tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Traitement des données", "Visualisations", "Modelisation", "Evaluation"])

with tabs_1:
    
    # Afficher le tableau
    st.title("Chargement du jeu de donnée")
    st.header("Affichage du tableau")
    st.write("Voici un aperçu du dataframe directement après son importation :")
    st.dataframe(df.head())
    st.write("Nous pouvons constater que la première colonne n'est pas été définie en tant qu'index.")
    
    st.header("Définition de l'index")
    st.markdown("Nous pouvons le faire en utilisant la commande `.set_index()`.")
    df.columns = ["index","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium","total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","od280/od315_of_diluted_wines","proline","target"]
    st.write("Voici un aperçu du dataframe directement après ce petit traitement :")
    df = df.set_index("index")

    # Afficher le tableau
    st.dataframe(df.head())
    
    # STATS DESCRIPTIVES
    st.title("Analyse descriptive du dataframe")
    
    st.header("Types des colonnes")
    st.dataframe(df.dtypes)
    
    st.header("Variables quantitatives")
    st.write(df.drop("target", axis=1).describe(include='all'))
    
    st.header("Variables qualitatives")
    st.write(df["target"].value_counts())

with tabs_2:    
    st.title("Variables qualitatives")
    st.header("target")
    st.bar_chart(df["target"].value_counts())

    st.title("Variables quantitatives")

    st.header("alcohol")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["alcohol"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="alcohol")
        
    #############################################################################################
    
    #############################################################################################
    
    st.header("malic_acid")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["malic_acid"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="malic_acid")
    
    st.header("ash")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["ash"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="ash")
    
    st.header("alcalinity_of_ash")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["alcalinity_of_ash"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="alcalinity_of_ash")
    
    st.header("magnesium")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["magnesium"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="magnesium")
    
    st.header("total_phenols")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["total_phenols"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="total_phenols")
    
    st.header("flavanoids")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["flavanoids"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="flavanoids")
    
    st.header("nonflavanoid_phenols")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["nonflavanoid_phenols"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="nonflavanoid_phenols")
    
    st.header("proanthocyanins")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["proanthocyanins"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="proanthocyanins")
    
    st.header("color_intensity")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["color_intensity"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="color_intensity")
    
    st.header("hue")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["hue"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="hue")
    
    st.header("od280/od315_of_diluted_wines")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["od280/od315_of_diluted_wines"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="od280/od315_of_diluted_wines")
    
    st.header("proline")
    col1, col2, col3 = st.columns(3)
    #
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(df["proline"], fill=True, color="skyblue", ax=ax)
        ax.set_title("Densité de la colonne 'alcohol'")
        ax.set_xlabel("Degré d'alcool")
        ax.set_ylabel('Densité')
        st.pyplot(fig)
    #
    with col2:
        st.scatter_chart(df, x="target", y="proline")
    

with tabs_3:
    pass

with tabs_4:
    pass