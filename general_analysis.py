import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def distrib_plots(data):
    data = data.iloc[:, 1:]  # Suppression de la première colonne si nécessaire
    option = st.selectbox(
        "Valeur à étudier ?", options=list(data.drop(columns='target').columns),
    )

    # Slidebar pour filtrer les degrés d'alcool
    if option in data.drop(columns='target').columns:
        # Paramètres interactifs
        min_val = round(np.min(data[option]))
        max_val = round(np.max(data[option]))
        bins = st.slider('Combien de bins ?', min_value=1, max_value=100, value=10)

        # Graphiques
        fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=True, constrained_layout=True)  # Taille augmentée + espacement
        cpt = 0
        for target in pd.unique(data['target']):
            temp = data[data['target'] == str(target)]
            hist, bins_edges = np.histogram(temp[option], bins=bins, range=[min_val, max_val])
            fig.suptitle(f'Distribution de {option} pour chaque type de vin', fontsize=14, weight='bold')
            label = f"{str(target)}"
            axs[cpt].hist(temp[option], bins_edges, histtype='stepfilled', label=label, alpha=0.7, edgecolor='black', color='#A13636')

            # Réglage explicite des étiquettes
            axs[cpt].set_xlabel(option, fontsize=12, weight='bold')
            axs[cpt].set_ylabel('Compte', fontsize=12, weight='bold')
            axs[cpt].tick_params(axis='both', which='major', labelsize=10)  # Taille des graduations
            axs[cpt].legend(loc='upper right', fontsize=10)
            axs[cpt].grid(True, linestyle='--', alpha=0.6)
            cpt += 1

        st.pyplot(fig)

        # Calcul des fréquences
        F = 100 * hist / (data.shape[0])
        bins_col = []
        for i in range(len(bins_edges) - 1):
            temp = f"[{bins_edges[i]:.2f}, {bins_edges[i + 1]:.2f}]"
            bins_col.append(temp)

        Fdict = {bins_col[i]: F[i] for i in range(len(bins_col))}
        freq_df = pd.DataFrame(Fdict, index=[0])
        st.write(f'Fréquences de {option} par bin en pourcentage')
        st.dataframe(freq_df)

def correlation_matrice(data):
    # Option pour afficher les corrélations entre les colonnes numériques
    if st.checkbox("Afficher les corrélations entre les colonnes numériques"):
        st.subheader("Matrice de correlation")
        # Vérification des colonnes numériques uniquement
        corr = data.iloc[:, 1:].select_dtypes(include=["float64", "int64"]).corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr[mask] = np.nan
        st.dataframe(
            corr.style
            .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
            .highlight_null(color='#f1f1f1')
            .format(precision=2)
        )

def pairplot(data):
    # Visualisation des relations entre les variables numériques (Pairplot)
    if st.checkbox("Afficher le pairplot des variables numériques"):
        # Vérification de la présence de colonnes numériques avant d'afficher le pairplot
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        if not numeric_data.empty:
            st.write("Visualisation des relations entre les variables numériques :")
            pairplot_fig = sns.pairplot(numeric_data)
            st.pyplot(pairplot_fig)
        else:
            st.write("Aucune colonne numérique disponible pour afficher le pairplot.")

    # Option pour sauvegarder les données transformées
    if st.checkbox("Sauvegarder les données transformées"):
        csv = data.to_csv(index=False)
        st.download_button(
            label="Télécharger les données transformées",
            data=csv,
            file_name="data_transformed.csv",
            mime="text/csv",
        )

def select_graphes(data):
    st.title("Variables qualitatives")
    st.header("target")
    st.bar_chart(data["target"].value_counts(), color='#A13636')

    st.title("Variables quantitatives")
    colonne_selection = st.selectbox("Sélectionnez une variable :", data.iloc[:, 1:].select_dtypes(include=["float64", "int64"]).columns, key="col_select")
    graphe_selection = st.selectbox("Sélectionnez un graphe :", ["Boxplot","Courbe de densité","Nuage de point"], 
                                    key="graph_select")
    
    st.header(f"Variable **{colonne_selection}**")
        
    if graphe_selection == "Boxplot":     
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x='target', y=colonne_selection, palette="Set2", ax=ax)
        ax.set_title(f"Boxplot de {colonne_selection} selon le Target")
        ax.set_xlabel("Target")
        ax.set_ylabel(colonne_selection)
        st.pyplot(fig)
        
    elif graphe_selection == "Courbe de densité": 
        fig, ax = plt.subplots()
        sns.kdeplot(data[colonne_selection], fill=True, color="#A13636", ax=ax)
        ax.set_title(f"Densité de la colonne {colonne_selection}")
        ax.set_xlabel(colonne_selection)
        ax.set_ylabel('Densité')
        st.pyplot(fig)

    elif graphe_selection == "Nuage de point":    
        # Vérifiez que les colonnes existent dans vos données
        if "target" in data.columns and colonne_selection in data.columns:
            fig = px.scatter(
                data,
                x="target",
                y=colonne_selection,
                color_discrete_sequence=["#A13636"]
            )
            st.plotly_chart(fig)
        else:
            st.error("Les colonnes 'target' ou la colonne sélectionnée n'existent pas dans les données.")