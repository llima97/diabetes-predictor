import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Inject Custom CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
    }
    h1 {
        color: #0B5394;
        font-size: 36px;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTextInput > div > div > input {
        border-radius: 5px;
        padding: 5px;
    }
    .stSelectbox > div > div > div {
        border-radius: 5px;
    }
    .stRadio > div {
        background-color: #e8f0fe;
        padding: 10px;
        border-radius: 8px;
    }
    .predict-box {
        background-color: #d9edf7;
        color: #31708f;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        text-align: center;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Load saved model, scaler, and columns ---
model = pickle.load(open("C:/Users/Windows/Desktop/diabetes_predictor/xgboost_model.pkl", "rb"))
scaler = pickle.load(open("C:/Users/Windows/Desktop/diabetes_predictor/scaler.pkl", "rb"))
columns = pickle.load(open("C:/Users/Windows/Desktop/diabetes_predictor/columns.pkl", "rb"))

# --- Streamlit App Layout ---
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction with SHAP Explanation")

st.markdown("#### Fill in the patient information below:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.radio("Hypertension?", ["No", "Yes"])
    smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)

with col2:
    age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
    heart_disease = st.radio("Heart Disease?", ["No", "Yes"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120, step=1)

# --- Predict Button ---
if st.button("🔍 Predict Now"):
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }])

    # One-hot encode like training
    input_encoded = pd.get_dummies(input_data)
    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[columns]

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    result = "Not Diabetic ☺️" if prediction == 0 else "Diabetic 🙁"
    
    st.markdown(f'<div class="predict-box"><b>Prediction Result:</b> {result}</div>', unsafe_allow_html=True)

    # SHAP explanation
    st.subheader("🔍 Why this prediction?")
    explainer = shap.Explainer(model)
    shap_values = explainer(pd.DataFrame(input_scaled, columns=columns))

    # Show SHAP waterfall
    shap.plots.waterfall(shap_values[0], show=False)
    import matplotlib.pyplot as plt
    st.pyplot(plt.gcf())




