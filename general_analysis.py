import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def distrib_plots(data):
    data = data.iloc[:,1:]
    option = st.selectbox(
    "Valeur à étudier ?", options = list(data.drop(columns='target').columns),
    )

    # slidebar pour filrer les ° d'alcool
    if option in data.drop(columns='target').columns :
        # distrib du degrée d'alcool parmis les vins
        plt.rcParams['font.size'] = '4'
        min = round(np.min(data[option]))
        max = round(np.max(data[option]))
        bins = st.slider('Combien de bins ?', min_value= 1,max_value=100)

    # Ici on pourrait peut être rendre ça interactif ? En demandant la range de °
        fig, axs = plt.subplots(1,3,figsize=(6,2),sharey=True)
        cpt = 0
        for target in pd.unique(data['target']) : 
            temp = data[data['target'] == str(target)]
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
        F = 100*hist/(data.shape[0])
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
