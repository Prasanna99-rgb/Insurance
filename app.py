import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Load Model
# -------------------------------
model_path = "Self.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Upload Self.pkl")
else:
    model = pickle.load(open(model_path, "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Insurance Prediction App", layout="centered")

st.title("💰 Insurance Cost Prediction")
st.write("Enter details below:")

# -------------------------------
# Input Fields (REAL FEATURES)
# -------------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

smoker = st.selectbox("Smoker", ["No", "Yes"])
smoker_yes = 1 if smoker == "Yes" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Insurance Cost"):
    try:
        features = np.array([[age, bmi, children, smoker_yes]])
        prediction = model.predict(features)

        st.success(f"💸 Estimated Cost: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
