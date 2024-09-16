import streamlit as st
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Utilisation d'un backend non interactif pour matplotlib
# plt.switch_backend('Agg')

# Configuration de l'URL de l'API
API_URL = "https://api-flask-credit-368cdcf3e0aa.herokuapp.com"

# Fonction pour obtenir la liste des IDs clients:
def get_client_ids():
    try:
        response = requests.get(f"{API_URL}/list_ids")
        response.raise_for_status()
        ids = response.text.strip('Liste des id client.e.s valides :\n\n').strip('[]').split(', ')
        return [int(id) for id in ids]
    except requests.RequestException as e:
        st.error(f"Erreur lors de la récupération des IDs client.e.s: {e}")
        return []

# Fonction pour obtenir les informations et prédictions d'un client
def get_client_prediction(client_id):
    response = requests.get(f"{API_URL}/prediction/{client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération des informations de l'individu.")
        return None

# Fonction pour obtenir le graphique SHAP global
def get_global_shap_plot():
    response = requests.get(f"{API_URL}/global_shap")
    if response.status_code == 200:
        img_data = response.text.split('base64,')[1]
        return Image.open(BytesIO(base64.b64decode(img_data)))
    else:
        st.error("Erreur lors de la récupération du graphique global SHAP.")
        return None

# Fonction pour obtenir le graphique SHAP local d'un client
def get_local_shap_plot(client_id):
    response = requests.get(f"{API_URL}/local_shap/{client_id}")
    if response.status_code == 200:
        img_data = response.text.split('base64,')[1]
        return Image.open(BytesIO(base64.b64decode(img_data)))
    else:
        st.error("Erreur lors de la récupération du graphique local SHAP.")
        return None

# Fonction pour obtenir les données pour comparaison
def get_client_data():
    try:
        response = requests.get(f"{API_URL}/client_data")
        response.raise_for_status()
        return pd.read_json(response.text)
    except requests.RequestException as e:
        st.error(f"Erreur lors de la récupération des données client.e.s: {e}")
        return pd.DataFrame()
    
# Fonction pour obtenir les données brutes pour comparaison :
def get_client_raw_data():
    try:
        response = requests.get(f"{API_URL}/client_raw_data")
        response.raise_for_status()
        return pd.read_json(response.text)
    except requests.RequestException as e:
        st.error(f"Erreur lors de la récupération des données brutes client.e.s: {e}")
        return pd.DataFrame()
    
# Fonction pour obtenir les données scalées pour comparaison :
    
def get_scaled_data():
    try:
        response = requests.get(f"{API_URL}/scaled_data")
        response.raise_for_status()
        return pd.read_json(response.text)
    except requests.RequestException as e:
        st.error(f"Erreur lors de la récupération des données scalées: {e}")
        return pd.DataFrame()
    
# Fonction pour afficher la jauge
def display_gauge(score, threshold, in_sidebar=False):
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=score,
        mode="gauge+number+delta",
        title={'text': "Score", 'font': {'size': 24}},
        delta={'reference': threshold, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "green" if score > threshold else "red"},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    if in_sidebar:
        fig.update_layout(autosize=False, width=300, height=250)
        st.sidebar.plotly_chart(fig)
    else:
        fig.update_layout(autosize=False, width=400, height=350)
        st.plotly_chart(fig)


# Configuration de la page d'accueil
def main():
    st.title("Tableau de bord - Décision de Crédit")
    st.header("Bienvenue sur le tableau de bord d'aide à la décision d'attribution de crédit.")

    # Afficher un avertissement, en rouge, pour rappeler que c'est un outil d'aide à la décision :
    st.write(":warning: *:red[Rappel : Ceci est un outil d'aide à la décision et ne remplace pas l'avis d'un professionnel.]*")

    # # Sélection client :

    client_ids = get_client_ids()
    if not client_ids:
        st.error("Impossible de récupérer les IDs client.e.s.")
        return
    client_id = st.selectbox('Veuillez sélectionnez un ID client.e', client_ids,
                         index=None, placeholder="Liste identifiants client.e.s")
    
    # Obtention de la prédiction :
    if st.button("Obtenir la prédiction"):
        prediction = get_client_prediction(client_id)

        if prediction:
            st.session_state.prediction = prediction
            st.session_state.show_analysis = False

    if 'prediction' in st.session_state:
        prediction = st.session_state.prediction
        if prediction['statut'] == 'Crédit accepté':
            st.success(":white_check_mark: Le crédit est accordé.")
        else:
            st.error(":x: Le crédit est refusé.")

        prob_defaut = round(prediction['probabilité_défaut'] * 100, 2)
        score = 1 - prediction['probabilité_défaut']
        threshold = 0.377
    
        st.write(f"**Nous estimons la probabilité de défaut à:** {prob_defaut}%")
        st.write(f'*Le seuil de refus est fixé à : {threshold * 100:.2f}% (obtenu lors de la modélisation)*')
        
        client_infos = prediction['client_infos']
        st.sidebar.write("**Informations de la personne sélectionnée :**")
        st.sidebar.write(f"*Sexe :* {client_infos['sexe']}")
        st.sidebar.write(f"*Âge :* {client_infos['âge']} ans")
        st.sidebar.write(f"*Revenu :* {client_infos['revenu']} $")
        st.sidebar.write(f"*Source de revenu :* {client_infos['source_revenu']}")
        st.sidebar.write(f"*Montant du crédit :* {client_infos['montant_credit']} $")
        st.sidebar.write(f"*Statut familial :* {client_infos['statut_famille']}")
        st.sidebar.write(f"*Éducation :* {client_infos['education']}")
        st.sidebar.write(f"*Ratio revenu/crédit :* {client_infos['ratio_revenu_credit']}%")

        if prediction['statut'] == 'Crédit accepté':
            statut = "Crédit accordé"
        else:
            statut = "Crédit refusé"        
        
        # Ajout des informations de probabilité de défaut et du score dans la sidebar
        st.sidebar.write("**Informations de prédiction :**")
        st.sidebar.write(f"*Statut du crédit :* {statut}.")
        st.sidebar.write(f"*Score :* {score:.2f}")

        # Affichage de la jauge du score dans la sidebar
        st.sidebar.write("*Jauge de score :*")
        display_gauge(score, 1 - threshold, in_sidebar=True)

        # st.sidebar.write(f"**Score :** {score:.2f}")
        # st.sidebar.write(f"**Seuil de refus :** {threshold}")

        # Affichage de la jauge du score
        st.write("**Jauge de score :**")
        display_gauge(score, 1 - threshold)

        if st.checkbox("Analyse des variables les plus importantes"):
            st.session_state.show_analysis = True

        if 'show_analysis' in st.session_state and st.session_state.show_analysis:
            st.write("**Top 10 des variables les plus importantes pour le modèle**")
            global_shap_plot = get_global_shap_plot()
            if global_shap_plot:
                st.image(global_shap_plot, caption="Feature Importance Globale SHAP")

            st.write("**Top 9 des variables les plus importantes pour l'individu sélectionné**")
            local_shap_plot = get_local_shap_plot(client_id)
            if local_shap_plot:
                st.image(local_shap_plot, caption=f"Feature Importance Locale SHAP pour l'individu {client_id}")
            
            # Explication SHAP :
            st.write("*L'explication SHAP (SHapley Additive exPlanations) est une méthode de décomposition des prédictions de modèles de machine learning.* "
                     "*Elle attribue une valeur d'importance à chaque variable pour chaque prédiction.* "
                     "*Les valeurs SHAP positives indiquent une contribution positive à la prédiction (probabilité de défaut plus élevée),* " 
                     "*tandis que les valeurs négatives indiquent une contribution négative (probabilité de défaut moins élevée).* "
                     "*Plus la valeur SHAP est élevée, plus la variable a un impact sur la prédiction.* "
                     "*Les graphiques ci-dessus montrent les 10 variables les plus importantes pour le modèle et les 9 variables les plus importantes pour l'individu sélectionné.*")

            if st.checkbox("Comparaison vs autres client.e.s (variables SHAP)"):
                client_data = get_client_data()
                client_raw_data = get_client_raw_data()

                if client_data.empty or client_raw_data.empty:
                    st.error("Erreur lors de la récupération des données de comparaison.")
                else:
                    # On s'assure que l'Id client est bien dans l'index
                    if client_id in client_data['SK_ID_CURR'].values:
                        selected_client = client_data[client_data['SK_ID_CURR'] == client_id]
                    else:
                        selected_client = pd.DataFrame()

                # Ajout de débogage
                # st.write("Données du client sélectionné :")
                # st.write(selected_client)

                top_10_features_shap = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DOWN_PAYMENT',
                                        'PAYMENT_RATE', 'EXT_SOURCE_1', 
                                        'BURO_NB_CURRENCY', 'PREV_PERC_INST_PAID_ON_TIME',
                                        'YEARS_EMPLOYED', 'MEAN_PREV_CNT_PAYMENT']

                feature = st.selectbox('Veuillez sélectionner une variable à comparer', top_10_features_shap,
                                    index=0, placeholder="Liste variables")

                if feature:
                    st.write(f"Pour l'individu {client_id}, la valeur de {feature} est {selected_client[feature].values[0].round(2)}. Moyenne pour l'ensemble des client.e.s : {client_data[feature].mean().round(2)}.")
                    st.write("*Les différences de valeurs pour une même variable entre ce graphique et le graphique précédent*\n"
                                "*sont dues à la normalisation des données nécessaire à la modélisation.*")
                    fig, ax = plt.subplots()
                    sns.kdeplot(client_data[feature], label='Ensemble client.e.s', ax=ax)

                    if not selected_client.empty:
                        # Vérifiez que la feature existe dans les données du client sélectionné
                        if feature in selected_client.columns:
                            # st.write(f"Valeur de {feature} pour le client sélectionné : {selected_client[feature].values[0]}")
                            ax.axvline(selected_client[feature].values[0], color='orange', label='Client.e sélectionné.e')
                        else:
                            st.warning(f"La feature {feature} n'existe pas pour l'individu sélectionné.")
                    else:
                        st.warning("L'individu sélectionné n'a pas été trouvé dans les données.")

                    ax.set_title(f'Distribution de la variable {feature}')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Densité')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.error("Veuillez sélectionner une variable.")

                # Comparaison sur d'autres variables :
                if st.checkbox("Comparaison sur d'autres variables"):
                    client_data = get_client_data()
                    client_raw_data = get_client_raw_data()
                    if client_data.empty or client_raw_data.empty:
                        st.error("Erreur lors de la récupération des données de comparaison.")
                    else:
                        # Assurez-vous que l'ID client est bien dans l'index
                        if client_id in client_raw_data['SK_ID_CURR'].values:
                            selected_client = client_raw_data[client_raw_data['SK_ID_CURR'] == client_id]
                        else:
                            selected_client = pd.DataFrame()

                    # Ajout de débogage
                    # st.write("Données du client sélectionné :")
                    # st.write(selected_client)

                    other_features = ['YEARS_BIRTH', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'NAME_INCOME_TYPE',
                                            'AMT_CREDIT', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']

                    other_feature = st.selectbox('Veuillez sélectionner une variable à comparer', other_features,
                                        index=0, placeholder="Liste variables")

                    if other_feature:
                        if client_raw_data[other_feature].nunique() > 10:
                            st.write(f"*Pour l'individu {client_id}, la valeur de {other_feature} est {selected_client[other_feature].values[0]}. Moyenne de l'ensemble des client.e.s : {client_raw_data[other_feature].mean().round(2)}.*")
                            fig, ax = plt.subplots()
                            sns.kdeplot(client_raw_data[other_feature], label='Ensemble client.e.s', ax=ax)

                            if not selected_client.empty:
                                # Vérifiez que la feature existe dans les données du client sélectionné
                                if other_feature in selected_client.columns:
                                    # st.write(f"Valeur de {feature} pour le client sélectionné : {selected_client[feature].values[0]}")
                                    ax.axvline(selected_client[other_feature].values[0], color='orange', label='Client.e sélectionné.e')
                                else:
                                    st.warning(f"La feature {other_feature} n'existe pas pour l'individu sélectionné.")
                            else:
                                st.warning("L'individu sélectionné n'a pas été trouvé dans les données.")

                            ax.set_title(f'Distribution de la variable {other_feature}')
                            ax.set_xlabel(other_feature)
                            ax.legend(loc='upper right')
                            st.pyplot(fig)
                        # Si la feature sélectionnée a moins de 10 valeurs uniques, affichez un pie chart où
                        # la modalité du client sélectionné est mise en évidence:
                        else:
                            explode = []
                            possible_values = client_raw_data[other_feature].value_counts().index.to_list()
                            client_value = selected_client[other_feature].values[0]
                            for value in possible_values:
                                if value == client_value:
                                    explode.append(0.1)
                                else:
                                    explode.append(0)
                            # st.write(f"Valeurs possibles : {possible_values}")
                            fig, ax = plt.subplots()
                            wedges, texts, autotexts = ax.pie(
                                client_raw_data[other_feature].value_counts(), autopct='%1.1f%%', explode=explode, startangle=90,
                                textprops=dict(color="w"), pctdistance=0.85
                            )
                            ax.legend(wedges, possible_values, title=other_feature, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

                            # Ajuster les étiquettes
                            for text in texts:
                                text.set_fontsize(8)
                            for autotext in autotexts:
                                autotext.set_fontsize(8)

                            # Dessiner un cercle au centre pour créer un effet de beignet
                            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                            fig.gca().add_artist(centre_circle)

                            ax.set_title(f"Répartition de la variable {other_feature}")
                            ax.set_ylabel('')
                            # ax.legend(title=other_feature, loc='upper right', bbox_to_anchor=(1, 0, 0.5, 1))
                            st.pyplot(fig)

                    else:
                        st.error("Veuillez sélectionner une variable.")

                    # J'aimerais maintenant ajouter une autre case qui permettrait de voir des corrélations, via scatterplot,
                    # entre une liste de variables choisies par l'utilisateur (choix restreint via une liste déroulante).
                    if st.checkbox("Corrélation entre variables"):
                        features_scaled = get_scaled_data()
                        st.write("**Sélectionnez deux variables pour afficher le scatterplot de leur corrélation**")
                        features_corr = ['CREDIT_INCOME_RATIO','EXT_SOURCE_2', 'AMT_INCOME_TOTAL',  'EXT_SOURCE_3', 'DOWN_PAYMENT', 'PAYMENT_RATE', 'EXT_SOURCE_1', 
                                    'BURO_NB_CURRENCY', 'PREV_PERC_INST_PAID_ON_TIME', 'YEARS_EMPLOYED', 
                                    'MEAN_PREV_CNT_PAYMENT','AMT_CREDIT', 'YEARS_BIRTH',
                                    'INCOME_PER_PERSON']
                        feature_x = st.selectbox('Variable X', features_corr, index=0, placeholder="Liste variables")
                        feature_y = st.selectbox('Variable Y', features_corr, index=1, placeholder="Liste variables")

                        if feature_x and feature_y:
                            fig, ax = plt.subplots()
                            sns.scatterplot(x=feature_x, y=feature_y, data=features_scaled, ax=ax)
                            ax.set_title(f"Corrélation entre {feature_x} et {feature_y}")
                            st.pyplot(fig)
                        else:
                            st.error("Veuillez sélectionner deux variables.")



if __name__ == "__main__":
    main()
