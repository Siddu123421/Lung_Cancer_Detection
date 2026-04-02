import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("lung_cancer_model (1).pkl")

st.title("Lung Cancer Detection App")

age = st.number_input("Age", 1, 100)
gender = st.selectbox("Gender (1=Male, 0=Female)", [1, 0])
smoking = st.selectbox("Smoking (1=Yes, 0=No)", [1, 0])
stress = st.selectbox("Mental Stress (1=Yes, 0=No)", [1, 0])
pollution = st.selectbox("Pollution Exposure (1=Yes, 0=No)", [1, 0])
yellow_fingers = st.selectbox("Yellow Fingers", [1, 0])
chronic_disease = st.selectbox("Chronic Disease", [1, 0])
fatigue = st.selectbox("Fatigue", [1, 0])
allergy = st.selectbox("Allergy", [1, 0])
alcohol = st.selectbox("Alcohol Consumption", [1, 0])
coughing = st.selectbox("Coughing", [1, 0])
shortness_of_breath = st.selectbox("Shortness of Breath", [1, 0])
chest_pain = st.selectbox("Chest Pain", [1, 0])

if st.button("Predict"):
    data = np.array([[
    age,                  # AGE
    gender,               # GENDER
    smoking,              # SMOKING
    yellow_fingers,       # FINGER_DISCOLORATION
    stress,               # MENTAL_STRESS
    pollution,            # EXPOSURE_TO_POLLUTION
    chronic_disease,      # LONG_TERM_ILLNESS
    fatigue,              # ENERGY_LEVEL
    allergy,              # IMMUNE_WEAKNESS
    shortness_of_breath,  # BREATHING_ISSUE
    alcohol,              # ALCOHOL_CONSUMPTION
    coughing,             # THROAT_DISCOMFORT
    95,                   # OXYGEN_SATURATION
    chest_pain,           # CHEST_TIGHTNESS
    1,                    # FAMILY_HISTORY
    1,                    # SMOKING_FAMILY_HISTORY
    1,                    # STRESS_IMMUNE
]])
    result = model.predict(data)
    proba = model.predict_proba(data)

    if result[0] == 1:
        st.error(f"High Risk ({proba[0][1]*100:.2f}%)")
    else:
        st.success(f"Low Risk ({proba[0][0]*100:.2f}%)")