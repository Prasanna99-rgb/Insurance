import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Insurance AI Predictor",
    page_icon="🏥",
    layout="centered"
)

# ---------------- Custom UI + Animation ----------------
st.markdown("""
<style>

/* Background Gradient Animation */
.stApp {
    background: linear-gradient(-45deg, #1f4037, #99f2c8, #4facfe, #00f2fe);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.15);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.02);
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}

/* Button Animation */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 20px rgba(0,114,255,0.7);
}

/* Result Glow */
.result {
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    color: #00ffcc;
    animation: glow 1.5s infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 10px #00ffcc; }
    to { text-shadow: 0 0 25px #00ffcc; }
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
try:
    model = pickle.load(open("model_pickle.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Model load error: {e}")
    st.stop()

# ---------------- Title ----------------
st.markdown('<div class="title">🏥 AI Insurance Predictor</div>', unsafe_allow_html=True)
st.write("")

# ---------------- Card Layout ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.number_input("BMI", 15.0, 50.0, 25.0)

with col2:
    children = st.slider("Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["No", "Yes"])

st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------- Prediction ----------------
if st.button("🚀 Predict Charges"):

    with st.spinner("Analyzing data... 🤖"):
        try:
            smoker_yes = 1 if smoker == "Yes" else 0

            input_data = pd.DataFrame(
                [[age, bmi, children, smoker_yes]],
                columns=['age', 'bmi', 'children', 'smoker_yes']
            )

            prediction = model.predict(input_data)[0]

            st.markdown(
                f'<div class="result">💰 ${prediction:,.2f}</div>',
                unsafe_allow_html=True
            )

            st.write("")

            # Insights
            if smoker == "Yes":
                st.warning("⚠️ Smoking increases insurance cost")
            else:
                st.success("✅ Non-smoker → lower cost")

            if bmi > 30:
                st.warning("⚠️ High BMI may increase cost")

            if age > 50:
                st.warning("⚠️ Higher age → higher charges")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<center style='color:white;'>🔥 AI Powered | Premium UI Design</center>",
    unsafe_allow_html=True
)
