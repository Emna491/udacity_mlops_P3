from sklearn.model_selection import train_test_split
from joblib import load

from src.ml.data import process_data
from src.ml.model import compute_model_metrics, inference

import pytest
import numpy as np
import pandas as pd

from .ml.data import process_data
from .ml.model import train_model

@pytest.fixture
def data():
    df = pd.read_csv("./data/census_revised.csv")
    return df


@pytest.fixture
def model():
    model = load("./model/model.joblib")

    return model


@pytest.fixture
def encoder():
    encoder = load("./model/encoder.joblib")
    return encoder


@pytest.fixture
def lb():
    lb = load("./model/lb.joblib")
    return lb


def test_null_values(data):
    assert data.shape == data.dropna().shape, "Dropping nan values will change the shape"

def test_process_data(data, encoder, lb):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert data.shape[0] == np.shape(X)[0], "The number of samples has changed"
    assert data.shape[0] == np.shape(y)[0], "The number of samples has changed"
    assert np.ndim(y) == 1, "Labels shape has "

    
def test_train_model(model, encoder,lb):

    train_model(data, cat_features, root_path='./')

    assert os.path.isfile("./model/model.joblib")
    assert os.path.isfile("./model/encoder.joblib")
    assert os.path.isfile("./model/lb.joblib")
