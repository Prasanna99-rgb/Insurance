import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Insurance Predictor", page_icon="💰", layout="wide")

# -------------------------------
# Background Image + Styling
# -------------------------------
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1450101499163-c8848c66ca85");
        background-size: cover;
    }
    .main {
        background-color: rgba(255,255,255,0.85);
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
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
st.sidebar.title("📌 About")
st.sidebar.info("Insurance cost prediction using ML")

st.sidebar.write("👨‍💻 Prasanna Deshmane")

# -------------------------------
# Title
# -------------------------------
st.title("💰 Insurance Cost Prediction")
st.write("Predict your medical insurance cost instantly")

# -------------------------------
# Inputs
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("👤 Age", 18, 100, 25)
    bmi = st.slider("⚖️ BMI", 10.0, 50.0, 25.0)

with col2:
    children = st.slider("👶 Children", 0, 5, 0)
    smoker = st.selectbox("🚬 Smoker", ["No", "Yes"])

smoker_yes = 1 if smoker == "Yes" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Cost"):
    features = np.array([[age, bmi, children, smoker_yes]])
    prediction = model.predict(features)[0]

    st.success(f"💸 Estimated Cost: ₹ {prediction:,.2f}")

    # Insight
    if smoker_yes == 1:
        st.warning("⚠️ Smoking increases cost")
    else:
        st.info("✅ Lower risk profile")

    # -------------------------------
    # Graph (Age vs Cost Trend)
    # -------------------------------
    st.subheader("📊 Cost vs Age Trend")

    ages = list(range(18, 65))
    costs = []

    for a in ages:
        pred = model.predict([[a, bmi, children, smoker_yes]])[0]
        costs.append(pred)

    fig, ax = plt.subplots()
    ax.plot(ages, costs)
    ax.set_xlabel("Age")
    ax.set_ylabel("Predicted Cost")
    ax.set_title("Insurance Cost Growth with Age")

    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | ML Deployment Project")
