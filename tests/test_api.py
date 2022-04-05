from fastapi.testclient import TestClient
import API.main
from API.main import app


def test_get():
    with TestClient(app) as client:
        response = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello world!"}



def test_negative_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": {"salary": "<=50K"}}


def test_prediction_positive_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "workclass": "Private",
                "education": "Masters",
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": {"salary": ">50K"}}


