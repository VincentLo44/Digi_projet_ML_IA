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
    with tabs_3:
      st.write('test colone')
      option = st.selectbox(
      "Valeur à étudier ?", options = list(df.drop(columns='target').columns),
      )
    #slidebar pour filrer les ° d'alcool ???

      if option in df.drop(columns='target').columns :
          #distrib du degrée d'alcool parmis les vins
          plt.rcParams['font.size'] = '4'
          min = round(np.min(df[option]))
          max = round(np.max(df[option]))
          bins = st.slider('Combien de bins ?', min_value= 1,max_value=100)

      #Ici on pourrait peut être rendre ça interactif ? En demandant la range de ° ?

          fig, axs = plt.subplots(1,3,figsize=(6,2),sharey=True)
          cpt = 0
          for target in pd.unique(df['target']) : 
              temp = df[df['target'] == str(target)]
              hist,bins = np.histogram(temp[option],bins=bins,range=[min,max])
              fig.suptitle('Distribution de %s pour chaque type de vin'%option,fontsize=7)
              label = "%s" % (str(target))
              axs[cpt].hist(temp[option],bins,histtype='stepfilled',label=label)
              axs[cpt].set_xlabel(option)
              axs[cpt].set_ylabel('Compte')
              axs[cpt].legend()
              cpt = cpt +1
          fig.align_ylabels([axs[0],axs[1]])
          fig.align_ylabels([axs[1],axs[2]])
          st.pyplot(fig)
          F = 100*hist/(df.shape[0])
          bins_col = []
          cpt = 0
          while True : 
              temp = '['+ str(bins[cpt])+','+str(bins[cpt+1])+']'
              bins_col.append(temp)
              cpt+=1
              if cpt == len(bins)-1 :
                  break
          cpt = 0 
          Fdict = {}
          while True : 
              Fdict[bins_col[cpt]] = F[cpt]
              cpt+=1 
              if cpt == len(bins)-1 :
                  break
          temp = pd.DataFrame(Fdict, index = np.arange(start=0,stop=1))
          st.write('Fréquences de %s par bin en pourcentage'%option)
          st.dataframe(temp)

if __name__ == '__main__':
    header()
