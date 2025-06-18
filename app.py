import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Title
st.title("🏦 Bank Customer Churn Prediction & Training")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    return data

data = load_data()

# # Hide raw data display
# if st.checkbox("Show raw data"):
#     st.info("🔒 Raw data is hidden for confidentiality purposes.")

# Select features to show in the input form
feature_cols = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 
                'products_number', 'credit_card', 'active_member', 'estimated_salary']

target_col = 'churn'  # Target column

# Sidebar: Select model
model_choice = st.sidebar.selectbox(
    "Choose Model to Train and Predict:",
    ("Decision Tree", "Random Forest", "XGBoost")
)

# Sidebar: Input form
st.sidebar.header("Input Customer Data")

def user_input_features():
    credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)
    country = st.sidebar.selectbox("Country", data['country'].unique())
    gender = st.sidebar.selectbox("Gender", data['gender'].unique())
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure = st.sidebar.slider("Tenure", 0, 10, 3)
    balance = st.sidebar.number_input("Balance", 0.0, float(data['balance'].max()), 50000.0)
    products_number = st.sidebar.slider("Number of Products", 1, 4, 1)
    credit_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
    active_member = st.sidebar.selectbox("Active Member", [0, 1])
    estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, float(data['estimated_salary'].max()), 60000.0)

    return pd.DataFrame({
        'credit_score': [credit_score],
        'country': [country],
        'gender': [gender],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary]
    })

input_df = user_input_features()

# Data Preprocessing function
def preprocess(df, train_df=None):
    # Combine with training data to get all dummies consistent
    if train_df is not None:
        combined = pd.concat([df, train_df], axis=0)
    else:
        combined = df.copy()

    combined = pd.get_dummies(combined, columns=['country', 'gender'], drop_first=True)
    
    # If training data provided, keep only train columns to align features
    if train_df is not None:
        combined = combined.reindex(columns=train_df.columns, fill_value=0)

    return combined.iloc[0:len(df)]

# Prepare training data
X = data[feature_cols]
y = data[target_col]

X_processed = preprocess(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model button
if st.sidebar.button("Train Model"):
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:  # XGBoost
        model = XGBClassifier(gamma=0.1, learning_rate=0.1, max_depth=3, n_estimators=100, use_label_encoder=False, eval_metric='logloss')

    model.fit(X_train, y_train)
    joblib.dump(model, f"{model_choice.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X_processed.columns, "model_features.pkl")
    st.success(f"{model_choice} model trained and saved successfully!")

# Load model for prediction
try:
    model = joblib.load(f"{model_choice.lower().replace(' ', '_')}_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_features = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.warning("Train the model first!")

# Predict button
if st.sidebar.button("Predict Churn"):
    if 'model' in locals():
        # Preprocess input data same as training data
        input_processed = preprocess(input_df, pd.DataFrame(columns=model_features))
        input_scaled = scaler.transform(input_processed)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.error(f"Prediction: Customer WILL churn. Probability: {proba:.2f}" if proba else "Prediction: Customer WILL churn.")
        else:
            st.success(f"Prediction: Customer will NOT churn. Probability: {proba:.2f}" if proba else "Prediction: Customer will NOT churn.")
    else:
        st.warning("Model not trained or loaded. Please train the model first.")
