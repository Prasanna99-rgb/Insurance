import streamlit as st
import pickle
import numpy as np

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Smart Prediction App", page_icon="🤖", layout="wide")

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("Self.pkl", "rb"))

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Input Parameters")

feature1 = st.sidebar.number_input("Feature 1", min_value=0.0)
feature2 = st.sidebar.number_input("Feature 2", min_value=0.0)
feature3 = st.sidebar.number_input("Feature 3", min_value=0.0)
feature4 = st.sidebar.number_input("Feature 4", min_value=0.0)

# ------------------ MAIN PAGE ------------------
st.title("🤖 Machine Learning Prediction App")
st.markdown("### Enter details in the sidebar and click predict")

# ------------------ BUTTON ------------------
if st.button("🚀 Predict"):

    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    prediction = model.predict(input_data)

    st.success(f"✅ Prediction Result: {prediction[0]}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
