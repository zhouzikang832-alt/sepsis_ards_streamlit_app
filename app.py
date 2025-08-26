import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Sepsis–ARDS Gut Dysbiosis Risk Predictor", layout="centered")

st.title("🧬 Sepsis–ARDS Gut Dysbiosis Risk Predictor")
st.caption("An online prediction tool powered by Python/Streamlit. Place your trained model `final_model.pkl` in the same folder.")

# --- Load model ---
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        with open("final_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, e

model, load_err = load_model()
if load_err:
    st.error(f"Model loading failed: {load_err}")
    st.stop()

# --- Input section ---
st.subheader("Enter Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=60, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
    spo2_max = st.number_input("SPO₂ Max (%)", min_value=50, max_value=100, value=95, step=1)

with col2:
    hr_min = st.number_input("HR-Min (bpm)", min_value=20, max_value=200, value=60, step=1)
    ph = st.number_input("Arterial pH", min_value=6.8, max_value=7.8, value=7.35, step=0.01, format="%.2f")
    abs_mono = st.number_input("Absolute monocyte count (×10⁹/L)", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    vent = st.selectbox("Mechanical ventilation", ["Yes", "No"])

# Assemble input DataFrame
X_in = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Race": race,
    "SPO2-MAX": spo2_max,
    "HR-MIN": hr_min,
    "PH": ph,
    "Absolute monocytes count": abs_mono,
    "Vent": vent,
}])

st.markdown("**Submitted Features:**")
st.dataframe(X_in)

# --- Prediction ---
st.subheader("Prediction")
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Run Prediction"):
    try:
        proba = model.predict_proba(X_in)[:, 1]
        y_pred = int(proba[0] >= threshold)

        st.success(f"Predicted probability of gut dysbiosis: {proba[0]:.2%}")
        if y_pred == 1:
            st.warning("⚠ High Risk: Consider early gut function assessment, nutritional optimization, probiotics/prebiotics, and close monitoring.")
        else:
            st.info("✅ Low Risk: Continue standard management and reassess dynamically.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
