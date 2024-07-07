import numpy as np
import pandas as pd
from scipy.stats import mode
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load datasets
data = pd.read_csv(r"C:\Users\Pradnya\OneDrive\OfficeMobile\Downloads\archive\Training.csv").dropna(axis=1)

# Preprocess data
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train models
svm_model = SVC()
nb_model = GaussianNB()
rf_model = RandomForestClassifier(random_state=18)

final_svm_model = svm_model.fit(X, y)
final_nb_model = nb_model.fit(X, y)
final_rf_model = rf_model.fit(X, y)

# Save the models and encoder
joblib.dump(final_svm_model, 'svm_model.pkl')
joblib.dump(final_nb_model, 'nb_model.pkl')
joblib.dump(final_rf_model, 'rf_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Create a dictionary for symptom index
symptoms = X.columns.values
symptom_index = { " ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms) }

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

def predict_disease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        #"final_prediction": final_prediction
    }
    return predictions
