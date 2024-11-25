import streamlit as st
import pandas as pd

def upload_data():
    st.sidebar.header("Chargez un fichier CSV")
    """Chargement du fichier CSV."""
    uploaded_file = st.sidebar.file_uploader("Chargez un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data_target_column = data.iloc[:, 1:]
        return data, data_target_column
    else:
        st.error("Veuillez charger un fichier CSV pour continuer.")
        return None, None