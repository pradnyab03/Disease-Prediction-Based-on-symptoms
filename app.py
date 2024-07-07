
from flask import Flask, request, render_template
import joblib 
import numpy as np
from collections import Counter

app = Flask(__name__)

# Load models and encoder
svm_model = joblib.load('svm_model.pkl')
nb_model = joblib.load('nb_model.pkl')
rf_model = joblib.load('rf_model.pkl')
encoder = joblib.load('encoder.pkl')

# Load symptom index and class names
symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
"chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
"muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue",
"weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss",
"restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
"high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
"indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
"loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain",
"diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure",
"fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision",
"phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose",
"congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
"pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
"cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
"puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger",
"extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain",
"muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements",
"loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
"foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
"depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
"belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", "increased_appetite",
"polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration",
"visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
"distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum", "prominent_veins_on_calf",
"palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
"skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
"red_sore_around_nose", "yellow_crust_ooze"

]

symptom_index = { " ".join([i.capitalize() for i in symptom.split("_")]): index for index, symptom in enumerate(symptoms) }

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form['symptoms']
        predictions = predict_disease(symptoms)
        return render_template('result.html', predictions=predictions)
    except Exception as e:
        return str(e)

def predict_disease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

    # Use Counter to find the most common prediction
    final_prediction = Counter([rf_prediction, nb_prediction, svm_prediction]).most_common(1)[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        #"final_predictrion": final_prediction
    }
    return predictions

if __name__ == '__main__':
    app.run(debug=True)
