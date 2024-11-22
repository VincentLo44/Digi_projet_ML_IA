import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Projet ML",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)
tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Traitement des données", "Visualisations", "Modelisation", "Evaluation"])

from modelisation import modelisation
from machine_learning import machine_learning

def upload_data():
    """Chargement du fichier CSV."""
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données :")
        data_target_column = data.iloc[:, 1:]
        st.write(data_target_column.head())
        return data, data_target_column
    else:
        st.warning("Veuillez charger un fichier CSV pour continuer.")
        return None, None

def header():
    # Chargement des données
    data, data_target_column = upload_data()
    if data is None or data_target_column is None:
        return  # Arrêtez si aucune donnée n'est chargée

    with tabs_1:
        modelisation(data, data_target_column)
    with tabs_2:
        machine_learning(data, data_target_column)


if __name__ == '__main__':
    header()