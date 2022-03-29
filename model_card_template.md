# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest Classifier is used with n_estimators=100, and the remaining parameters wer set to their default values.
## Intended Use
The model  predicts whether income exceeds $50K/yr based on census data.
## Training Data
The data used from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).
Data consist of 14 featuress, where 6 categorical attributes have been considered (workclass, education, marital-status, occupation, relationship, race, sex, native-country).
## Evaluation Data
The evaluation data is a 20% split fom the whole data
## Metrics
The precision recall and F1-score were used to assess the model performance 

## Ethical Considerations
The model discriminates on race, gender and origin country.
## Caveats and Recommendations
Nwe data and more advances models should be  tested.