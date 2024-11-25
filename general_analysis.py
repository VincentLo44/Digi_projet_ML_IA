import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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
            axs[cpt].hist(temp[option], bins_edges, histtype='stepfilled', label=label, alpha=0.7, edgecolor='black')

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
