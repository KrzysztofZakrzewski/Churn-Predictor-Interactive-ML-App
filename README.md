## 📊 Churn Predictor – Interactive ML App
## 🚀 Overview

This is a simple interactive application that predicts customer churn risk based on user-provided features.

The app is built using a trained Logistic Regression model and allows real-time testing of how different customer attributes influence churn probability.

## 🔍 Features
Predicts churn probability based on customer data
Uses Logistic Regression with L1 regularization
Applies a custom decision threshold optimized for recall
Fully interactive interface built with Streamlit
## 🧠 Model
Algorithm: Logistic Regression (L1)
ROC AUC: ~0.86
Recall (churn): ~0.68
Focus: interpretability and business-driven threshold tuning
## 🖥️ Demo

👉 Fill in customer details to see:

churn probability
final prediction (Churn / No Churn)
## ⚙️ How to Run Locally
conda env create -f environment.yml
conda activate churn-env
streamlit run app.py
📂 Project Structure
```
.
├── app.py
├── environment.yml
├── clasyfication_model
|        ├── churn_model_final.pkl
|        ├── churn_model.pkl
|
├── Telco_Customer_clasyfication_ML.ipynb
```

Telco Customer Churn dataset
(used for training and evaluation)

🧩 Notes
The model expects preprocessed input (handled internally in the app)
Features are aligned with the training dataset using consistent encoding
🔗 Links
🌐 Portfolio: [https://krzysztofzakrzewski.github.io/portfolio/]
💼 LinkedIn: [https://www.linkedin.com/in/krzysztof-zakrzewski-206554258/]
📦 EDA project: [https://krzysztofzakrzewski.github.io/portfolio/Telco_Customer_clasyfication_EDA/]
📦 ML project: [https://krzysztofzakrzewski.github.io/portfolio/Telco_Customer_clasyfication_ML/]

This app was built to demonstrate how machine learning models can be:

interpretable
interactive
aligned with business decisions
🔥 One-liner

ML is not just about models — it's about decisions.


## 📓 Notebook

A notebook is included for experimenting with the model and testing different scenarios.