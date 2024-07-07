import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv(r"C:\Users\Pradnya\OneDrive\OfficeMobile\Downloads\archive\Training.csv").dropna(axis=1)

# Preprocess the data
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train the models
svm_model = SVC()
nb_model = GaussianNB()
rf_model = RandomForestClassifier(random_state=18)

svm_model.fit(X, y)
nb_model.fit(X, y)
rf_model.fit(X, y)

# Save the models and encoder
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Models and encoder saved successfully!")
