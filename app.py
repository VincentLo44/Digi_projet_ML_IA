import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modelisation import modelisation
from machine_learning import machine_learning
from general_analysis import distrib_plots, correlation_matrice, pairplot, select_graphes
from upload_data import upload_data
from preexistant import upload_model_existant

# CONFIIIIIIIG bash
# streamlit run app.py --global.configFile=.streamlit/config.toml
# python -m streamlit run app.py --server.fileWatcherType=none

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Projet ML",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

tabs_1, tabs_2, tabs_3, tabs_4, tabs_5 = st.tabs(["Prévisualisation des données", "Analyse généralisée", 
                                          "Machine Learning", "Evaluation", "Modèle préexistant"])

def header():
    # Chargement des données
    data, data_target_column = upload_data()
    if data is None or data_target_column is None:
        return  # Arrêtez si aucune donnée n'est chargée

    with tabs_1:
        modelisation(data, data_target_column)
    with tabs_2:
        distrib_plots(data)
        correlation_matrice(data)
        pairplot(data)
        select_graphes(data)
    with tabs_3:
        machine_learning(data, data_target_column)
    with tabs_4:
        upload_model_existant(data, data_target_column)

    # with tabs_5:

if __name__ == '__main__':
    header()
