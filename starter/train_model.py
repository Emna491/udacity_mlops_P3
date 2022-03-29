# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

from joblib import dump
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

import pandas as pd
import json


def compute_slice_metrics(
    model, data, encoder, lb, cat_features, sliced_feature, label):

    dict_result = {}

    for i in data[sliced_feature].unique():
        data_slice = data[data[sliced_feature] == i]

        X, y, _, _ = process_data(
            data_slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        dict_result[i] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "sample": len(y),
        }

    return {sliced_feature: dict_result}

if __name__ == "main":
# Add code to load in the data.
    data = pd.read_csv("../data/census_revised.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    
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
    
    #Train data preprocessing
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    # Test data preprocessing
    X_test, y_test, _, _ = process_data(
            test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
    
    # Train and save model    
    model = train_model(X_train, y_train)
    
    dump(model, "../model/model.joblib")
    dump(encoder, "../model/encoder.joblib")
    dump(lb, "../model/lb.joblib")
    
    
    # Performance on slices
    metrics_slices = compute_slice_metrics(
        model, test, encoder, lb, cat_features, "education", "salary"
    )
    with open("../results/slice_output.json", "w") as fp:
        json.dump(metrics_slices, fp)
