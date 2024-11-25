import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def modelisation(data, data_target_column):
    st.title("Analyse de données avec Streamlit")
    st.write("Aperçu des données :")
    st.write(data_target_column.head())
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
