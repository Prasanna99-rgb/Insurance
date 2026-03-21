import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("Self.pkl", "rb"))

st.set_page_config(page_title="Prediction App", page_icon="📊")

st.title("📊 ML Prediction App")

# Get number of features
n_features = model.n_features_in_

st.write(f"🔢 Model expects {n_features} input features")

# Dynamic inputs
inputs = []

for i in range(n_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

# Prediction
if st.button("Predict"):
    input_data = np.array([inputs])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"✅ Prediction: {prediction}")
