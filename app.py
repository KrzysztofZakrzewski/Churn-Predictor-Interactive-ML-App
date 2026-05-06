# Imports

import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd

import joblib

st.title('Customer Retention Model')
st.write('Predicts customer churn risk based on key features.')
st.write('➡️ Fill in customer details to predict churn risk.')

loaded = joblib.load("clasyfication_model/churn_model_final.pkl")

model = loaded["model"]
threshold = loaded["threshold"]
features = loaded["features"]

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges", 0.0, 150.0, 50.0)

if st.button("Predict"):



    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges
    }

    df = pd.DataFrame([input_dict])
    
    
    
    df = pd.get_dummies(df)

    # dodaj brakujące kolumny
    for col in features:
        if col not in df:
            df[col] = 0

    df = df[features]
    

    
    proba = model.predict_proba(df)[:, 1][0]

    prediction = "Churn" if proba >= threshold else "No Churn"


    st.write(f"Probability of churn: {proba:.2f}")
    st.write(f"Prediction: {prediction}")

with st.expander("📊 Most influential features"):
    st.image("features_affecting_churn.png")


with st.expander("Credits"):
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("🔗 [View author project portfolio](https://krzysztofzakrzewski.github.io/portfolio/)")
    st.markdown("**GitHub:** [Repository](https://github.com/KrzysztofZakrzewski/Churn-Predictor-Interactive-ML-App)")
    st.markdown("**linkedin:** [Repository](https://www.linkedin.com/in/krzysztof-zakrzewski-206554258/)")

    st.markdown("**Orginal Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/code/emineyetm/telco-customer-churn/notebook)")