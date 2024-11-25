import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import joblib

def upload_model_existant(data, data_target_column):
    # Partie ajoutée : Chargement d'un modèle .pkl
    st.subheader("Charger un modèle préexistant")
    # Identification de la cible (target)
    target_column = st.selectbox("Sélectionnez la colonne cible", 
                                 data_target_column.select_dtypes(include=["float64", "int64"]).columns, 
                                 key="selectbox")
    features = [col for col in data.select_dtypes(include=["float64", "int64"]).columns if col != target_column]
    y = data[target_column]

    uploaded_model = st.file_uploader("Chargez un fichier modèle (.pkl)", type=["pkl"])

    if uploaded_model is not None:
        # Charger le modèle
        loaded_model = joblib.load(uploaded_model)
        st.success("Modèle chargé avec succès !")

        # Afficher les informations sur le modèle
        st.write("Modèle chargé :", type(loaded_model))

        # Demander à l'utilisateur d'entrer des données pour prédire
        input_data = data.iloc[:, 1:]
        for col in input_data.columns:
            input_data[col] = st.text_input(f"Valeur pour {col}", "")
        
        if st.button("Faire une prédiction avec le modèle chargé"):
            # Vérifier que toutes les valeurs sont présentes
            missing_cols = [col for col in input_data.columns if input_data[col].eq("").any()]
            if missing_cols:
                st.error(f"Certaines colonnes manquent des valeurs : {missing_cols}")
            else:
                # Convertir en DataFrame
                input_df = input_data
                
                # Corriger les types si nécessaire
                for col in features:
                    if input_df[col].dtype == object:
                        try:
                            input_df[col] = pd.to_numeric(input_df[col])
                        except ValueError:
                            pass
                
                # Prédiction avec le modèle chargé
                prediction = loaded_model.predict(input_df)
                st.write("Prédiction :", prediction)