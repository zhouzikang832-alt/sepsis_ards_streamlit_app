# app.py (dynamic feature UI + robust alignment)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import traceback
import re
from typing import List, Tuple, Dict

st.set_page_config(page_title="Sepsis–ARDS Gut Dysbiosis Risk Predictor", layout="centered")
st.title("🧬 Sepsis–ARDS Gut Dysbiosis Risk Predictor")
st.caption("This app inspects your model's expected features and builds input fields automatically. Upload final_model.pkl to the app folder.")

# -------------------------
# Helper functions
# -------------------------
def load_model(path="final_model.pkl"):
    """Try joblib then pickle."""
    try:
        mdl = joblib.load(path)
        return mdl, None
    except Exception as e_joblib:
        try:
            with open(path, "rb") as f:
                mdl = pickle.load(f)
            return mdl, None
        except Exception as e_pickle:
            tb = traceback.format_exc()
            return None, f"joblib error: {e_joblib}\n\npickle error: {e_pickle}\n\nTraceback:\n{tb}"

def unwrap_model(obj):
    """If a list/tuple was saved, find first element that has .predict"""
    if isinstance(obj, (list, tuple)):
        for elt in obj:
            if hasattr(elt, "predict"):
                return elt
        return None
    if hasattr(obj, "predict"):
        return obj
    return None

def get_expected_feature_names(model_obj) -> List[str]:
    """Try to get expected feature names from model or pipeline."""
    if hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)
    # If pipeline, try to find preprocessor or final estimator
    if hasattr(model_obj, "named_steps"):
        # sometimes the preprocessor stores the names, or final estimator has feature_names_in_
        for name, step in model_obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
        # try last estimator
        try:
            last = list(model_obj.named_steps.items())[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
        except Exception:
            pass
    return None

def group_onehot_like(cols: List[str]) -> Dict[str, List[str]]:
    """
    If many cols share prefix separated by '_' (e.g. Sex_Male, Sex_Female),
    return groups: prefix -> list of full columns.
    """
    groups = {}
    for c in cols:
        if "_" in c:
            prefix = c.split("_")[0]
            groups.setdefault(prefix, []).append(c)
    # only keep groups that have >1 member
    return {k: v for k, v in groups.items() if len(v) > 1}

def default_value_for_feature(name: str):
    """Heuristic default values for common clinical features."""
    s = name.lower()
    if "age" in s:
        return 60
    if "sex" in s or "gender" in s:
        return "Male"
    if "race" in s:
        return "White"
    if "spo2" in s:
        return 95
    if "ph" in s:
        return 7.35
    if "hr" in s or "heart" in s:
        return 60
    if "lac" in s or "lactate" in s:
        return 2.0
    if "wbc" in s:
        return 12.0
    if "cre" in s or "cr" in s:
        return 1.0
    if "album" in s:
        return 3.5
    if "plate" in s or "plt" in s:
        return 200
    if "bun" in s:
        return 14
    if "na" == s or s.endswith("_na") or "natrium" in s:
        return 140
    if "k" == s or s.endswith("_k"):
        return 4.0
    if "vent" in s or "mechan" in s:
        return "No"
    # fallback numeric
    return 0.0

def infer_input_widgets(expected: List[str]):
    """
    Build UI inputs for the expected features.
    Returns a dict: feature_name -> value (raw), and also returns the final DataFrame matching expected.
    """
    st.write("### Model expects these features:")
    st.write(expected)

    # Detect onehot-like groups
    onehot_groups = group_onehot_like(expected)  # prefix -> list of columns

    # Build UI for grouped (one-hot) features
    chosen_values = {}  # group_prefix -> choice
    for prefix, full_cols in onehot_groups.items():
        # derive labels from suffixes
        labels = [c[len(prefix)+1:] for c in full_cols]
        # clean labels
        labels = [lab.replace("_", " ").replace("-", " ").strip() for lab in labels]
        choice = st.selectbox(f"{prefix} (choose one)", options=labels, index=0)
        chosen_values[prefix] = choice

    # Build UI for remaining features (not handled as one-hot group)
    remaining = [c for c in expected if not any(c in cols for cols in onehot_groups.values())]
    values = {}  # final raw mapping for remaining expected columns
    # Build a per-base UI: if column is plain categorical like 'Sex' present selectbox
    for feat in remaining:
        low = feat.lower()
        # skip if already covered (should not happen)
        if feat in sum(list(onehot_groups.values()), []):
            continue
        # common categorical heuristics
        if any(k in low for k in ["sex", "gender"]):
            val = st.selectbox(f"{feat}", options=["Male", "Female", "Other", "Unknown"], index=0)
            values[feat] = val
        elif "race" in low:
            val = st.selectbox(f"{feat}", options=["White", "Black", "Asian", "Hispanic", "Other", "Unknown"], index=0)
            values[feat] = val
        elif any(k in low for k in ["vent", "mechan", "intub", "mv"]):
            val = st.selectbox(f"{feat}", options=["No", "Yes"], index=0)
            values[feat] = 1 if val == "Yes" else 0
        elif any(k in low for k in ["ne","epi","da","dobu","milri","vaso","press"]):
            # vasoactive drugs: assume numeric 0/1 if named as a flag in training
            val = st.selectbox(f"{feat}", options=["No", "Yes"], index=0)
            values[feat] = 1 if val == "Yes" else 0
        else:
            # numeric by default: use heuristic default
            d = default_value_for_feature(feat)
            if isinstance(d, str):
                # fallback to text input
                txt = st.text_input(f"{feat}", value=d)
                values[feat] = txt
            else:
                # numeric input
                step = 0.01 if ("ph" in feat.lower() or "ratio" in feat.lower()) else 1.0
                # choose range heuristics
                v = st.number_input(f"{feat}", value=float(d), step=step)
                values[feat] = v

    # Build final DataFrame matching expected columns:
    final_row = {}
    for feat in expected:
        # if feat is part of a one-hot group, set 1/0
        matched_group = None
        for prefix, cols in onehot_groups.items():
            if feat in cols:
                matched_group = prefix
                break
        if matched_group is not None:
            chosen = chosen_values[matched_group]
            suffix = feat[len(matched_group)+1:].replace("_", " ").replace("-", " ").strip()
            # If selected label equals suffix -> 1 else 0
            final_row[feat] = 1 if suffix == chosen else 0
            continue

        # otherwise from values map
        if feat in values:
            final_row[feat] = values[feat]
        else:
            # last resort: try case-insensitive matches in 'values'
            found = False
            for k, v in values.items():
                if k.lower() == feat.lower():
                    final_row[feat] = v
                    found = True
                    break
            if not found:
                # fallback default
                final_row[feat] = default_value_for_feature(feat)

    final_df = pd.DataFrame([final_row], columns=expected)
    return final_df

# -------------------------
# Load model
# -------------------------
model_raw, load_err = load_model("final_model.pkl")
if load_err:
    st.error("Failed to load model:\n\n" + load_err)
    st.stop()

# handle numpy array saved by mistake
if isinstance(model_raw, np.ndarray):
    st.error("final_model.pkl contains a numpy.ndarray (predictions), not a model.\nSave the model object with joblib.dump(model, 'final_model.pkl').")
    st.stop()

model = unwrap_model(model_raw)
if model is None:
    st.error(f"Loaded object is not a model (type={type(model_raw)}). Save the estimator object.")
    st.stop()

st.success(f"Model loaded: {type(model)}")

# -------------------------
# Provide CSV upload for one-row inputs (optional)
# -------------------------
st.markdown("### Or upload a one-row CSV matching the model's expected columns")
uploaded = st.file_uploader("Upload CSV (one row)", type=["csv"])
uploaded_df = None
if uploaded:
    try:
        uploaded_df = pd.read_csv(uploaded)
        st.write("Uploaded dataframe preview:")
        st.dataframe(uploaded_df.head())
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

# -------------------------
# Determine expected features
# -------------------------
expected = get_expected_feature_names(model)
if expected is None:
    st.warning("Could not auto-detect model.feature_names_in_. You can either (A) upload a one-row CSV with the exact feature columns used during training, or (B) manually provide inputs by creating a CSV locally and uploading it.")
    # allow user to provide CSV
    if uploaded_df is not None:
        X_in = uploaded_df
    else:
        st.stop()
else:
    # If user uploaded CSV, validate and use it if it matches expected (case-insensitive)
    if uploaded_df is not None:
        # align columns case-insensitively
        lc_map = {c.lower(): c for c in uploaded_df.columns}
        expected_lc = [c.lower() for c in expected]
        if all(e in lc_map for e in expected_lc):
            # reorder columns into expected order
            ordered = [lc_map[e] for e in expected_lc]
            X_in = uploaded_df[ordered].copy()
        else:
            st.warning("Uploaded CSV does not contain all expected columns. Falling back to interactive inputs below.")
            X_in = infer_input_widgets(expected)
    else:
        # Build interactive inputs dynamically
        X_in = infer_input_widgets(expected)

# Final alignment: ensure columns match expected names and order
# If X_in has case differences, reorder
if isinstance(X_in, pd.DataFrame):
    # map columns case-insensitively if needed
    col_map = {c.lower(): c for c in X_in.columns}
    ordered_cols = []
    for ec in expected:
        if ec in X_in.columns:
            ordered_cols.append(ec)
        elif ec.lower() in col_map:
            ordered_cols.append(col_map[ec.lower()])
        else:
            st.error(f"Feature mismatch: Model expects column '{ec}' but it was not provided. Provided columns: {list(X_in.columns)}")
            st.stop()
    # reorder and rename to exactly expected names if there were case differences
    X_in = X_in[ordered_cols]
    rename_map = {ordered_cols[i]: expected[i] for i in range(len(expected))}
    X_in = X_in.rename(columns=rename_map)
else:
    st.error("Internal error: constructed input is not a DataFrame.")
    st.stop()

st.markdown("**Final input being sent to model (ordered to match model):**")
st.dataframe(X_in)

# -------------------------
# Prediction
# -------------------------
threshold = st.slider("Probability threshold", 0.0, 1.0, 0.95, 0.01)

if st.button("Predict"):
    try:
        # Try predict_proba, fallback to decision_function or predict
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_in)
            if proba.ndim == 1:
                pos = proba
            elif proba.shape[1] >= 2:
                pos = proba[:, 1]
            else:
                pos = proba[:, 0]
            prob = float(pos[0])
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_in)
            prob = float(1.0 / (1.0 + np.exp(-float(score[0]))))
        else:
            pred = model.predict(X_in)
            prob = float(pred[0])

        y_pred = int(prob >= threshold)
        st.success(f"Predicted probability (gut dysbiosis): {prob:.2%}")
        if y_pred == 1:
            st.warning("⚠ HIGH RISK: consider early gut monitoring/intervention (nutrition, probiotics, etc.)")
        else:
            st.info("✅ LOW RISK: routine management; reassess as needed.")
    except Exception as e:
        st.error(f"Prediction failed: {e}\n\nTraceback:\n{traceback.format_exc()}")
