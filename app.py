import streamlit as st
import numpy as np
import joblib
from utils import download_model

# Download and load the model
download_model()
model = joblib.load("models/heart_disease_rf_optimized.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction App")

st.markdown("Provide the patient's details in the sidebar to predict heart disease risk.")

# Sidebar Inputs
st.sidebar.header("🧾 Input Patient Data")

age = st.sidebar.slider("Age", 18, 100, 40)
gender = st.sidebar.radio("Gender", ["Female", "Male"])
height = st.sidebar.slider("Height (cm)", 100, 250, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 200, 70)
ap_hi = st.sidebar.slider("Systolic BP", 90, 200, 120)
ap_lo = st.sidebar.slider("Diastolic BP", 60, 140, 80)
cholesterol = st.sidebar.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
gluc = st.sidebar.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
smoke = st.sidebar.radio("Do you smoke?", ["No", "Yes"])
alco = st.sidebar.radio("Consume alcohol?", ["No", "Yes"])
active = st.sidebar.radio("Physically Active?", ["No", "Yes"])

# Mapping inputs
gender = 1 if gender == "Female" else 2
cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
cholesterol = cholesterol_map[cholesterol]
gluc = gluc_map[gluc]
smoke = int(smoke == "Yes")
alco = int(alco == "Yes")
active = int(active == "Yes")

# Feature engineering
bmi = weight / ((height / 100) ** 2)
hypertension = int(ap_hi >= 140 or ap_lo >= 90)
pulse_pressure = ap_hi - ap_lo

age_group_MidAge = int(30 < age <= 45)
age_group_Old = int(45 < age <= 60)
age_group_VeryOld = int(age > 60)
cholesterol_2 = int(cholesterol == 2)
cholesterol_3 = int(cholesterol == 3)
gluc_2 = int(gluc == 2)
gluc_3 = int(gluc == 3)

input_features = np.array([[age, gender, height, weight, ap_hi, ap_lo, smoke, alco, active,
                            bmi, hypertension, pulse_pressure,
                            cholesterol_2, cholesterol_3, gluc_2, gluc_3,
                            age_group_MidAge, age_group_Old, age_group_VeryOld]])

if st.sidebar.button("🚀 Predict"):
    prediction = model.predict(input_features)[0]
    prob = model.predict_proba(input_features)[0][1]

    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"🔴 High Risk of Heart Disease. Probability: **{prob:.2f}**")
        st.markdown("### 🔔 Recommendations:")
        st.markdown("""
        - 📆 Schedule a heart health checkup soon.
        - 🧂 Reduce salt and avoid fried/junk food.
        - 🚶 Increase physical activity and manage weight.
        - 🧘‍♂️ Manage stress and get quality sleep.
        - ❗ Consult a cardiologist immediately.
        """)
    else:
        st.success(f"🟢 Low Risk of Heart Disease. Probability: **{prob:.2f}**")
        st.markdown("### ✅ Tips to Stay Healthy:")
        st.markdown("""
        - 🥗 Maintain a balanced diet.
        - 🚴 Exercise regularly (at least 30 minutes/day).
        - 🚭 Avoid smoking and excessive alcohol.
        - 📊 Track blood pressure and cholesterol levels.
        - 💤 Ensure 7–8 hours of sleep daily.
        """)

st.markdown("---")
st.markdown("📌 *This app uses a machine learning model for risk prediction. For medical concerns, always consult a doctor.*")
