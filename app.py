import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---- LOAD MODEL ----
model = pickle.load(open("model_pickle.pkl", "rb"))

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰", layout="wide")

# ---- TITLE ----
st.title("💰 Insurance Cost Prediction App")
st.markdown("Predict medical insurance charges based on user details")

# ---- SIDEBAR INPUT ----
st.sidebar.header("🧾 Enter User Details")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ---- ENCODING ----
sex_val = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0

# One-hot encoding for region
region_dict = {
    "northeast": [1, 0, 0, 0],
    "northwest": [0, 1, 0, 0],
    "southeast": [0, 0, 1, 0],
    "southwest": [0, 0, 0, 1]
}

region_vals = region_dict[region]

# ---- FINAL INPUT ----
input_data = np.array([[age, sex_val, bmi, children, smoker_val] + region_vals])

# ---- PREDICT ----
if st.button("🚀 Predict Insurance Cost", use_container_width=True):

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:#e3f2fd;
            text-align:center;">
            <h2>💰 Estimated Insurance Cost</h2>
            <h1 style="color:#1565c0;">₹ {prediction:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- FOOTER ----
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Insurance ML App")
