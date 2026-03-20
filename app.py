import streamlit as st
import pickle
import numpy as np
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Insurance Prediction App", page_icon="💰", layout="wide")

# ---------------- LOAD MODEL ----------------
model_path = "Skl.pkl"

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# ---------------- TITLE ----------------
st.title("💰 Insurance Cost Prediction App")
st.markdown("### Predict insurance charges based on user details")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Enter User Details")

age = st.sidebar.slider("Age", 18, 100, 25)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 22.5)
children = st.sidebar.slider("Number of Children", 0, 5, 0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ---------------- ENCODING ----------------
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

# Simple encoding for region
region_dict = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}
region = region_dict[region]

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Insurance Cost"):

    try:
        input_data = np.array([[age, sex, bmi, children, smoker, region]])

        prediction = model.predict(input_data)

        st.success(f"💰 Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")

    except Exception as e:
        st.error("❌ Error during prediction. Check model input format.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit | Insurance ML App")
