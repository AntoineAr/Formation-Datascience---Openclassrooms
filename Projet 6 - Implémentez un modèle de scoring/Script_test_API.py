"""
Il s'agit ici du fichier de test unitaire pour le projet. Il teste les fonctions du fichier main.py.
L'idée est ensuite d'intégrer ces tests dans le workflow de déploiement continu via Github Actions.
"""


# Import des packages nécessaires :

from main import load_data, load_scaler_model_explainer, prepare_data, get_clients_ids
import pandas as pd
import pickle
import pytest
import sklearn.preprocessing
import lightgbm
import sklearn.calibration
import requests
import shap

# Fonction qui teste si les données ne sont pas vides :
def not_empty_returns():
    data = load_data("./data/subset_train.csv")
    scaler, model, explainer = load_scaler_model_explainer()
    features = prepare_data(data, scaler)
    clients_ids = get_clients_ids(data)
    assert type(scaler) == sklearn.preprocessing.StandardScaler
    assert type(model) == lightgbm.sklearn.LGBMClassifier
    assert type(explainer) == shap.TreeExplainer
    assert data.shape != (0, 0)
    assert features.shape != (0, 0)
    assert len(clients_ids) != 0

# Fonction qui test que l'index du dataframe renvoyé par la fonction load_data est bien l'identifiant client :
def test_index_is_client_id():
    data = load_data("./data/subset_train.csv")
    assert data.index.name == 'SK_ID_CURR'

# Fonction qui teste quelques ID clients pour vérifier qu'ils sont bien valides :
client_id_test = [146124, 242167, 343897]
def test_ids_client():
    df = load_data("./data/subset_train.csv")
    scaler, model, explainer = load_scaler_model_explainer()
    features = prepare_data(df, scaler)
    clients_ids = get_clients_ids(features)
    for client_id in client_id_test:
        assert client_id in clients_ids

# Fonction qui teste le bon fonctionnement d'une prédiction pour un client à risque :
def test_prediction_client_risk():
    # URL de l'app :
    url = 'https://application-credit-7ba79bc598e5.herokuapp.com/prediction/'
    # ID client connu pour être à risque
    client_id_risk = 343897
    url_client = url + str(client_id_risk)
    # Envoi d'une requête GET avec l'ID client
    response = requests.get(url_client)
    # Vérifier que la requête a réussi (code de statut HTTP 200)
    assert response.status_code == 200
    # Analyser la réponse JSON
    response_json = response.json()
    # Vérifier que la réponse indique que le client est à risque
    assert response_json['statut'] == 'à risque'

# Fonction qui teste le bon fonctionnement d'une prédiction pour un client non risqué :
def test_prediction_client_no_risk():
    # URL de l'app :
    url = 'https://application-credit-7ba79bc598e5.herokuapp.com/prediction/'
    # ID client connu pour ne pas être à risque
    client_id_no_risk = 374271
    url_client = url + str(client_id_no_risk)
    # Envoi d'une requête GET avec l'ID client
    response = requests.get(url_client)
    # Vérifier que la requête a réussi (code de statut HTTP 200)
    assert response.status_code == 200
    # Analyser la réponse JSON
    response_json = response.json()
    # Vérifier que la réponse indique que le client n'est pas à risque
    assert response_json['statut'] == 'non risqué'


