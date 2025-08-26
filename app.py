# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import traceback

st.set_page_config(page_title="Sepsis–ARDS Gut Dysbiosis Risk Predictor", layout="centered")

st.title("🧬 Sepsis–ARDS Gut Dysbiosis Risk Predictor")
st.caption("Upload your trained model as `final_model.pkl` in the app folder. This app will try joblib.load first, then pickle.load.")

# -------------------------
# Model loading utilities
# -------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str = "final_model.pkl"):
    """
    Try to load a model with joblib first (recommended for sklearn objects),
    then try pickle. Return (model_obj, None) on success or (None, error_message) on failure.
    """
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e_joblib:
        # fallback to pickle
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model, None
        except Exception as e_pickle:
            tb = traceback.format_exc()
            return None, f"Failed to load model. joblib error: {e_joblib}; pickle error: {e_pickle}\n{tb}"

def unwrap_model(obj):
    """
    If a tuple/list was saved, try to find the first element with a predict method.
    Otherwise return obj if it has predict, or None if no usable model is found.
    """
    if isinstance(obj, (list, tuple)):
        for elt in obj:
            if hasattr(elt, "predict"):
                return elt
        return None
    if hasattr(obj, "predict"):
        return obj
    return None

def get_expected_feature_names(model_obj):
    """
    Try to obtain expected feature names from model or pipeline.
    Returns a list of names or None.
    """
    # direct attribute
    if hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)
    # pipeline: try to locate final estimator
    if hasattr(model_obj, "named_steps"):
        try:
            # get last step
            last_step = list(model_obj.named_steps.items())[-1][1]
            if hasattr(last_step, "feature_names_in_"):
                return list(last_step.feature_names_in_)
        except Exception:
            pass
    return None

# -------------------------
# Load model
# -------------------------
model_raw, load_error = load_model("final_model.pkl")

if load_error:
    st.error("Model failed to load.\n\n" + load_error)
    st.stop()

# If loaded object is a numpy array -> likely saved predictions rather than model
if isinstance(model_raw, np.ndarray):
    st.error(
        "The file final_model.pkl contains a numpy.ndarray (likely saved predictions), not a model object.\n\n"
        "Please re-save your trained model object using e.g.:\n"
        "  joblib.dump(trained_model, 'final_model.pkl')\n\n"
        "Then upload the new final_model.pkl and redeploy."
    )
    st.stop()

# If object is tuple/list or something, unwrap to actual model
model = unwrap_model(model_raw)
if model is None:
    st.error(
        "The loaded object is not a model (no .predict found).\n\n"
        f"Loaded object type: {type(model_raw)}\n\n"
        "If you saved multiple objects, please save the fitted estimator (e.g. joblib.dump(model, 'final_model.pkl'))."
    )
    st.stop()

st.success(f"Model loaded successfully. Model type: {type(model)}")

# -------------------------
# Input panel
# -------------------------
st.subheader("Input patient parameters")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=60)
    spo2_max = st.number_input("SPO₂ max (%)", min_value=0, max_value=100, value=95)
    hr_min = st.number_input("HR min (bpm)", min_value=0, max_value=300, value=60)

with col2:
    ph = st.number_input("Arterial pH", min_value=6.8, max_value=7.8, value=7.35, format="%.2f")
    abs_mono = st.number_input("Absolute monocytes (×10⁹/L)", min_value=0.0, max_value=50.0, value=0.5, format="%.2f")
    vent = st.selectbox("Mechanical ventilation?", ["No", "Yes"])

# Build DataFrame for a single patient - update these column names to match your trained model
user_input = pd.DataFrame([{
    "age": age,
    "spo2_max": spo2_max,
    "ph": ph,
    "absolute_monocytes": abs_mono,
    "vent": 1 if vent == "Yes" else 0,
    "hr_min": hr_min
}])

st.markdown("**Submitted features:**")
st.dataframe(user_input)

# -------------------------
# Prediction helpers
# -------------------------
threshold = st.slider("Probability threshold (suggested: 0.95)", 0.0, 1.0, 0.95, 0.01)

def align_input_columns(input_df: pd.DataFrame, expected_cols):
    """
    Try to align input columns with expected_cols using case-insensitive matching.
    If some expected columns are missing, return (None, error_message).
    """
    provided = list(input_df.columns)
    lc_provided = {c.lower(): c for c in provided}
    reordered = []
    for col in expected_cols:
        if col in input_df.columns:
            reordered.append(col)
        elif col.lower() in lc_provided:
            reordered.append(lc_provided[col.lower()])
        else:
            return None, f"Model expects column '{col}' but it was not provided. Provided columns: {provided}"
    return input_df[reordered].copy(), None

# -------------------------
# Prediction action
# -------------------------
if st.button("Predict"):
    try:
        # Get expected feature names if available and align
        expected = get_expected_feature_names(model)
        if expected is not None:
            X_in, err = align_input_columns(user_input, expected)
            if err:
                st.error("Feature mismatch: " + err)
                st.stop()
        else:
            # If model does not expose expected names, use the user_input as is (assume correct order)
            X_in = user_input

        # Ensure DataFrame (many models accept DataFrame or 2D numpy)
        if not isinstance(X_in, pd.DataFrame):
            X_in = pd.DataFrame(X_in)

        # Prediction probability if possible
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_in)
            # handle cases where predict_proba returns shape (n,) or (n,1) or (n,2)
            if proba.ndim == 1:
                pos_prob = proba
            elif proba.shape[1] >= 2:
                pos_prob = proba[:, 1]
            else:
                pos_prob = proba[:, 0]
            prob = float(pos_prob[0])
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_in)
            # convert score to probability with sigmoid
            prob = float(1.0 / (1.0 + np.exp(-float(score[0]))))
        else:
            # fallback: use predict labels (0/1)
            pred_label = model.predict(X_in)
            prob = float(pred_label[0])
        
        predicted_label = int(prob >= threshold)

        st.success(f"Predicted probability (gut dysbiosis): {prob:.2%}")
        if predicted_label == 1:
            st.warning("⚠ HIGH RISK: Consider early gut monitoring/intervention (nutrition, probiotics, etc.).")
        else:
            st.info("✅ LOW RISK: Continue routine care and reassess as needed.")

    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Prediction failed: {e}\n\nTraceback:\n{tb}")
        st.stop()

st.markdown(
    "**Mechanistic background (gut–lung axis):** Sepsis-related gut barrier disruption and dysbiosis can aggravate pulmonary inflammation via the gut–lung axis, thereby worsening ARDS. Early identification of gut dysbiosis risk allows clinicians to intervene earlier."
)
