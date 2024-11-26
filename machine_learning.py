import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import seaborn as sns
import joblib
import io

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

        # Sauvegarde du modèle dans un objet en mémoire
        model_filename = "model.pkl"
        model_bytes = io.BytesIO()  # Utiliser BytesIO pour enregistrer en mémoire
        joblib.dump(pipeline, model_bytes)
        model_bytes.seek(0)  # Revenir au début du fichier en mémoire

        # Afficher un bouton de téléchargement
        st.download_button(
            label="Télécharger le modèle entraîné",
            data=model_bytes,
            file_name=model_filename,
            mime="application/octet-stream"
        )

        # Affichage des performances
        st.subheader("Résultats du Modèle")

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Précision du modèle : {accuracy * 100:.2f}%")

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            st.write("Matrice de confusion :")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu_r", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
            ax.set_xlabel("Prédictions")
            ax.set_ylabel("Réel")
            st.pyplot(fig)

            # Courbe ROC (uniquement pour les problèmes binaires ou adaptation pour multi-classes)
            if hasattr(model, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)
                
                if len(np.unique(y)) == 2:  # Cas binaire
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=np.unique(y)[1])
                    auc001 = roc_auc_score(y_test, y_proba[:, 1])
                    st.write(f"AUC : {auc001:.2f}")
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {auc001:.2f}")
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel("Taux de faux positifs")
                    ax.set_ylabel("Taux de vrais positifs")
                    ax.set_title("Courbe ROC")
                    ax.legend()
                    st.pyplot(fig)
                else:  # Cas multi-classes
                    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
                    auc002 = roc_auc_score(y_test_binarized, y_proba, multi_class="ovr")
                    st.write(f"AUC (multi-classes) : {auc002:.2f}")
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
