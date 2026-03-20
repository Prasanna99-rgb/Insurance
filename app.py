import streamlit as st
import pickle
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Prediction App", page_icon="🤖", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("Self.pkl", "rb"))

# (Optional) Load scaler if you have one
# scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🤖 AI Prediction Web App")
st.markdown("### Enter details and get instant prediction")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Input Features")

# 👉 Replace these with your real feature names
feature1 = st.sidebar.number_input("Age", 0, 100, 25)
feature2 = st.sidebar.number_input("Salary", 0, 100000, 50000)
feature3 = st.sidebar.number_input("Experience (Years)", 0, 50, 2)
feature4 = st.sidebar.number_input("Score", 0, 100, 60)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict Now"):

    input_data = np.array([[feature1, feature2, feature3, feature4]])

    # If using scaler
    # input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"✅ Prediction: {prediction[0]}")

    # If classification model → show probability
    try:
        prob = model.predict_proba(input_data)
        st.info(f"📊 Confidence: {np.max(prob)*100:.2f}%")
    except:
        pass

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built with Streamlit | By Prasanna")
