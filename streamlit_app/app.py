# streamlit_app/app.py
import streamlit as st
import pickle
import numpy as np
import os

# ─────────────────────────────
# Page configuration
# ─────────────────────────────
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🏥",
    layout="centered"
)

# ─────────────────────────────
# Load model
# ─────────────────────────────
@st.cache_resource   # loads model once — cached in memory
def load_model():
    model_path = '../malaria_api/model/malaria_model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# ─────────────────────────────
# App Header
# ─────────────────────────────
st.title("🏥 Malaria Detection System")
st.markdown("Enter patient symptoms below to get an instant diagnosis.")
st.divider()

# ─────────────────────────────
# Input Form
# ─────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age         = st.number_input("Patient Age",
                    min_value=1, max_value=120, value=25)
    fever       = st.selectbox("Fever?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No")
    chills      = st.selectbox("Chills?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    headache    = st.selectbox("Headache?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No")
    temperature = st.slider("Body Temperature (°C)",
                    min_value=35.0, max_value=45.0,
                    value=37.0, step=0.1)

st.divider()

# ─────────────────────────────
# Prediction Button
# ─────────────────────────────
if st.button("🔍 Run Diagnosis", use_container_width=True):

    # Prepare features
    features = np.array([[age, fever, chills, headache, temperature]])

    # Get prediction
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence  = max(probability) * 100
    risk        = ('HIGH'   if probability[1] > 0.7
                   else 'MEDIUM' if probability[1] > 0.4
                   else 'LOW')

    st.divider()

    # ─────────────────────────────
    # Display Results
    # ─────────────────────────────
    if prediction == 1:
        st.error("⚠️ MALARIA DETECTED")
    else:
        st.success("✅ NO MALARIA DETECTED")

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Diagnosis",   "Positive" if prediction == 1 else "Negative")
    m2.metric("Confidence",  f"{confidence:.1f}%")
    m3.metric("Risk Level",  risk)

    # Probability bar
    st.subheader("Probability Breakdown")
    st.progress(int(probability[1] * 100))
    st.caption(f"Malaria probability: {probability[1]*100:.1f}%")

    # Warning for high risk
    if risk == 'HIGH':
        st.warning("⚠️ High risk detected. "
                   "Please refer patient for lab confirmation immediately.")