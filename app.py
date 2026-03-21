import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Insurance Predictor", page_icon="💰", layout="wide")

# -------------------------------
# Custom CSS (Attractive UI)
# -------------------------------
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
    .stNumberInput, .stSelectbox {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
model_path = "Self.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Upload Self.pkl")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 About App")
st.sidebar.info(
    "This app predicts insurance cost based on user inputs using a Machine Learning model."
)

st.sidebar.write("### 👨‍💻 Developed by")
st.sidebar.write("Prasanna Deshmane")

# -------------------------------
# Main Title
# -------------------------------
st.title("💰 Insurance Cost Prediction App")
st.write("Fill the details below to estimate insurance cost")

# -------------------------------
# Layout (2 Columns)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)

with col2:
    children = st.number_input("👶 Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("🚬 Smoker", ["No", "Yes"])

smoker_yes = 1 if smoker == "Yes" else 0

# -------------------------------
# Prediction Button
# -------------------------------
st.markdown("---")

if st.button("🔍 Predict Insurance Cost"):
    try:
        features = np.array([[age, bmi, children, smoker_yes]])
        prediction = model.predict(features)

        st.success(f"💸 Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")

        # Extra Insight
        if smoker_yes == 1:
            st.warning("⚠️ Smoking increases insurance cost significantly!")
        else:
            st.info("✅ Non-smoker gets lower insurance cost.")

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("📊 Machine Learning Model Deployment using Streamlit")
