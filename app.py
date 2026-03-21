import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Insurance AI App", layout="wide")

# -------------------------------
# Background Styling
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1450101499163-c8848c66ca85");
    background-size: cover;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Login System
# -------------------------------
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# -------------------------------
# Load Model
# -------------------------------
model_path = "Self.pkl"

if not os.path.exists(model_path):
    st.error("Model not found!")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("💰 Insurance Prediction App")
    st.write("Welcome! Use this AI app to predict insurance costs.")

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Prediction":

    st.title("📊 Predict Insurance Cost")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 25)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)

    with col2:
        children = st.slider("Children", 0, 5, 0)
        smoker = st.selectbox("Smoker", ["No", "Yes"])

    smoker_yes = 1 if smoker == "Yes" else 0

    if st.button("Predict"):
        features = np.array([[age, bmi, children, smoker_yes]])
        prediction = model.predict(features)[0]

        st.success(f"💸 Cost: ₹ {prediction:,.2f}")

        # Graph
        st.subheader("📈 Age vs Cost")

        ages = list(range(18, 65))
        costs = [model.predict([[a, bmi, children, smoker_yes]])[0] for a in ages]

        fig, ax = plt.subplots()
        ax.plot(ages, costs)
        ax.set_xlabel("Age")
        ax.set_ylabel("Cost")

        st.pyplot(fig)

        # -------------------------------
        # Generate PDF Report
        # -------------------------------
        def create_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph(f"Insurance Cost Report", styles["Title"]))
            content.append(Paragraph(f"Age: {age}", styles["Normal"]))
            content.append(Paragraph(f"BMI: {bmi}", styles["Normal"]))
            content.append(Paragraph(f"Children: {children}", styles["Normal"]))
            content.append(Paragraph(f"Smoker: {smoker}", styles["Normal"]))
            content.append(Paragraph(f"Predicted Cost: ₹ {prediction:,.2f}", styles["Normal"]))

            doc.build(content)

        create_pdf()

        with open("report.pdf", "rb") as f:
            st.download_button("📥 Download Report", f, file_name="Insurance_Report.pdf")

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.title("ℹ️ About")
    st.write("This project uses Machine Learning to predict insurance costs.")
    st.write("Developed by Prasanna Deshmane")
