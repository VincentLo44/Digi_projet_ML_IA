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
import joblib

def upload_model_existant(data, data_target_column):
    # Partie ajoutée : Chargement d'un modèle .pkl
    st.subheader("Charger un modèle préexistant")
    
    # Identification de la cible (target)
    target_column = st.selectbox("Sélectionnez la colonne cible", 
                                 data_target_column.iloc[:, 1:].select_dtypes(include=["float64", "int64"]).columns, 
                                 key="selectbox")
    features = [col for col in data.iloc[:, 1:].select_dtypes(include=["float64", "int64"]).columns if col != target_column]
    y = data[target_column]

    # Charger un modèle existant
    uploaded_model = st.file_uploader("Chargez un fichier modèle (.pkl)", type=["pkl"])

    if uploaded_model is not None:
        # Charger le modèle
        loaded_model = joblib.load(uploaded_model)
        st.success("Modèle chargé avec succès !")

        # Afficher les informations sur le modèle
        st.write("Modèle chargé :", type(loaded_model))

        # Initialisation des valeurs moyennes pour les colonnes
        column_means = data[features].mean()

        # Interface utilisateur pour entrer des données (par défaut initialisé à la moyenne)
        st.subheader("Modifier les valeurs des colonnes pour faire une prédiction")
        input_data = {}
        for col in features:
            input_data[col] = st.number_input(f"Valeur pour {col}", value=column_means[col])

        # Convertir les données en DataFrame
        input_data = pd.DataFrame([input_data])

        # Vérification des colonnes manquantes
        required_columns = set(loaded_model.feature_names_in_)  # Colonnes attendues par le modèle
        input_columns = set(input_data.columns)

        missing_cols = required_columns - input_columns
        if missing_cols:
            # Ajouter les colonnes manquantes avec une valeur par défaut
            for col in missing_cols:
                if col == "Unnamed: 0":
                    input_data[col] = 0  # Valeur par défaut pour la colonne fictive
                else:
                    input_data[col] = 0  # Ajout d'autres colonnes manquantes avec une valeur neutre

        # Bouton pour effectuer la prédiction
        if st.button("Faire une prédiction avec le modèle chargé"):
            try:
                # Prédiction avec le modèle chargé
                prediction = loaded_model.predict(input_data)
                st.write("Prédiction :", prediction)
            except Exception as e:
                st.error(f"Erreur pendant la prédiction : {e}")
