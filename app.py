import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
 
# Afficher un graphique
#fig, ax = plt.subplots()
#ax.plot(data["Année"], data["Ventes"], marker="o")
#ax.set_title("Ventes par année")
#ax.set_xlabel("Année")
#ax.set_ylabel("Ventes")
#st.pyplot(fig)
 
 
 
st.set_page_config(
    page_title="Projet ML",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Traitement des données", "Visualisations", "Modelisation", "Evaluation"])
 
with tabs_1:
   
    # Import dataset
    st.title("Chargement du jeu de donnée")
    df = pd.read_csv("vin.csv",index_col=0) #on définit la 1e colonne comme index
    st.write("Le data set est composé de",df.shape[0],"lignes et",df.shape[1],"colones.")
   
    # Afficher le tableau
    st.header("Affichage du tableau")
    st.write("Voici un aperçu du dataframe directement après son importation :")
    st.write("Nous pouvons constater que la première colonne n'est pas été définie en tant qu'index.")
   
    st.header("Définition de l'index")
    st.markdown("Nous pouvons le faire en utilisant la commande `.set_index()`.")
    st.write("Voici un aperçu du dataframe directement après ce petit traitement :")
 
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
    pass
 
with tabs_3:
    pass
 
with tabs_4:
    pass

#slidebar pour filrer les ° d'alcool ???

#distrib du degrée d'alcool parmis les vins
min = round(np.min(df['alcohol']))
max = round(np.max(df['alcohol']))
bins = (max-min)*2

#Ici on pourrait peut être rendre ça interactif ? En demandant la range de ° ?

fig, axs = plt.subplots(1,3,figsize=(20, 10),sharey=True)

cpt = 0
for target in pd.unique(df['target']) : 
    temp = df[df['target'] == str(target)]
    hist,bins = np.histogram(temp['alcohol'],bins=bins,range=[min,max])
    fig.suptitle("Distribution du degré d'alcool pour les différents types de vins",fontsize=13)
    label = "%s" % (str(target))
    axs[cpt].hist(temp['alcohol'],bins,histtype='bar',label=label)
    axs[cpt].set_xlabel("Degrée d'alcool")
    axs[cpt].set_ylabel('Compte')
    axs[cpt].legend()
    cpt = cpt +1

fig.align_ylabels([axs[0],axs[1]])
fig.align_ylabels([axs[1],axs[2]])