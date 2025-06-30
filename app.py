
import streamlit as st
import numpy as np
import joblib
from utils import download_model

# Download the model first
download_model()

# Load the model
model = joblib.load("models/heart_disease_rf_optimized.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")

st.markdown("Fill in the following information to predict the likelihood of heart disease.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (in cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (in kg)", min_value=10, max_value=300, value=70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
    alco = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
    active = st.selectbox("Are you physically active?", ["No", "Yes"])
    submit = st.form_submit_button("Predict")

if submit:
    gender_val = 1 if gender == "Female" else 2
    cholesterol_val = ["Normal", "Above Normal", "Well Above Normal"].index(cholesterol) + 1
    gluc_val = ["Normal", "Above Normal", "Well Above Normal"].index(gluc) + 1
    smoke_val = 1 if smoke == "Yes" else 0
    alco_val = 1 if alco == "Yes" else 0
    active_val = 1 if active == "Yes" else 0

    bmi = weight / ((height / 100) ** 2)
    hypertension = int(ap_hi >= 140 or ap_lo >= 90)
    pulse_pressure = ap_hi - ap_lo

    age_group_MidAge = int(30 < age <= 45)
    age_group_Old = int(45 < age <= 60)
    age_group_VeryOld = int(age > 60)

    cholesterol_2 = int(cholesterol_val == 2)
    cholesterol_3 = int(cholesterol_val == 3)
    gluc_2 = int(gluc_val == 2)
    gluc_3 = int(gluc_val == 3)

    input_features = np.array([[age, gender_val, height, weight, ap_hi, ap_lo, smoke_val, alco_val, active_val,
                                bmi, hypertension, pulse_pressure,
                                cholesterol_2, cholesterol_3, gluc_2, gluc_3,
                                age_group_MidAge, age_group_Old, age_group_VeryOld]])

    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.error(f"🔴 High likelihood of heart disease.\nProbability: {round(probability, 2)}")
        st.info("👉 Suggestion: Please consult a cardiologist. Maintain a healthy diet, monitor blood pressure, and exercise regularly.")
    else:
        st.success(f"🟢 Low likelihood of heart disease.\nProbability: {round(probability, 2)}")
        st.info("✅ Suggestion: Keep up your healthy lifestyle! Regular check-ups are still recommended.")
