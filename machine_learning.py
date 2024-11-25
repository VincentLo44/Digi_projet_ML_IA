import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import joblib

def machine_learning(data, data_target_column):
    # Titre de l'application Streamlit
    st.title("Pipeline de Machine Learning avec Visualisations")

    # Identification de la cible (target)
    target_column = st.selectbox("Sélectionnez la colonne cible", data_target_column.columns)
    features = [col for col in data.columns if col != target_column]
    y = data[target_column]

    # Vérification du type de problème : classification ou régression
    is_classification = y.dtype == 'object' or len(y.unique()) < 20

    if is_classification:
        model_choice = st.selectbox(
            "Choisissez un algorithme de classification",
            ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
        )
    else:
        model_choice = st.selectbox(
            "Choisissez un algorithme de régression",
            ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor (SVR)"]
        )

    # Séparation des données
    X = data[features]
    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.select_dtypes(include=["object"]).columns

    # Pipelines pour le prétraitement
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns)
    ])

    # Modèle choisi
    if is_classification:
        model = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine (SVM)": SVC(probability=True)
        }[model_choice]
    else:
        model = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Regressor (SVR)": SVR()
        }[model_choice]

    # Pipeline complet
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Ajouter un bouton pour entraîner le modèle
    if st.button("Lancer l'entraînement"):
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entraînement du modèle
        pipeline.fit(X_train, y_train)
        st.success("Modèle entraîné avec succès !")

        # Prédictions
        y_pred = pipeline.predict(X_test)

        # Sauvegarde du modèle
        model_filename = "model.pkl"
        joblib.dump(pipeline, model_filename)
        st.success(f"Le modèle a été sauvegardé sous le nom : {model_filename}")

        # Affichage des performances
        st.subheader("Résultats du Modèle")

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Précision du modèle : {accuracy * 100:.2f}%")

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            st.write("Matrice de confusion :")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
            ax.set_xlabel("Prédictions")
            ax.set_ylabel("Réel")
            st.pyplot(fig)

            # Courbe ROC
            if hasattr(model, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=np.unique(y)[1])
                auc = roc_auc_score(y_test, y_proba)
                st.write(f"AUC : {auc:.2f}")
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("Taux de faux positifs")
                ax.set_ylabel("Taux de vrais positifs")
                ax.set_title("Courbe ROC")
                ax.legend()
                st.pyplot(fig)

        else:
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")

            # Graphique des prédictions vs réels
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Valeurs réelles")
            ax.set_ylabel("Valeurs prédites")
            ax.set_title("Valeurs réelles vs prédites")
            st.pyplot(fig)

    # Partie ajoutée : Chargement d'un modèle .pkl
    st.subheader("Charger un modèle préexistant")

    uploaded_model = st.file_uploader("Chargez un fichier modèle (.pkl)", type=["pkl"])

    if uploaded_model is not None:
        # Charger le modèle
        loaded_model = joblib.load(uploaded_model)
        st.success("Modèle chargé avec succès !")

        # Afficher les informations sur le modèle
        st.write("Modèle chargé :", type(loaded_model))

        # Demander à l'utilisateur d'entrer des données pour prédire
        input_data = data.iloc[:, 1:]
        for col in input_data:
            input_data[col] = st.text_input(f"Valeur pour {col}", "")
        
        if st.button("Faire une prédiction avec le modèle chargé"):
            # Vérifier que toutes les valeurs sont présentes
            missing_cols = [col for col in input_data.columns if input_data[col].eq("").any()]
            if missing_cols:
                st.error(f"Certaines colonnes manquent des valeurs : {missing_cols}")
            else:
                # Convertir en DataFrame
                input_df = pd.DataFrame([input_data])
                
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