import streamlit as st
import pandas as pd
import pickle
import joblib  # 添加joblib作为备选

st.set_page_config(page_title="Sepsis–ARDS Gut Dysbiosis Risk Predictor", layout="centered")

st.title("🧬 Sepsis–ARDS Gut Dysbiosis Risk Predictor")
st.caption("An online prediction tool powered by Python/Streamlit. Place your trained model `final_model.pkl` in the same folder.")

# --- Load model ---
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # 首先尝试用pickle加载
        try:
            with open("final_model.pkl", "rb") as f:
                model = pickle.load(f)
        except:
            # 如果pickle失败，尝试用joblib加载
            model = joblib.load("final_model.pkl")
        
        # 验证模型是否有predict_proba方法
        if hasattr(model, 'predict_proba'):
            return model, None
        else:
            return None, "Loaded object is not a valid model (missing predict_proba method)"
    except Exception as e:
        return None, f"Model loading failed: {e}"

model, load_err = load_model()
if load_err:
    st.error(load_err)
    st.stop()

# --- 确保输入特征与训练时一致 ---
# 根据你的建模代码，这些应该是模型期望的特征
expected_features = [
    "Age", "Sex", "Race", "SPO2-MAX", "HR-MIN", "PH", 
    "Absolute monocytes count", "Vent"
]

st.subheader("Enter Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=60, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
    spo2_max = st.number_input("SPO₂ Max (%)", min_value=50, max_value=100, value=95, step=1)

with col2:
    hr_min = st.number_input("HR-MIN (bpm)", min_value=20, max_value=200, value=60, step=1)
    ph = st.number_input("Arterial pH", min_value=6.8, max_value=7.8, value=7.35, step=0.01, format="%.2f")
    abs_mono = st.number_input("Absolute monocyte count (×10⁹/L)", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    vent = st.selectbox("Mechanical ventilation", ["Yes", "No"])

# 转换分类变量为数值（如果模型需要）
sex_mapping = {"Male": 1, "Female": 0}
race_mapping = {"White": 0, "Black": 1, "Asian": 2, "Other": 3}
vent_mapping = {"Yes": 1, "No": 0}

# Assemble input DataFrame with correct feature names
X_in = pd.DataFrame([{
    "Age": age,
    "Sex": sex_mapping[sex],
    "Race": race_mapping[race],
    "SPO2-MAX": spo2_max,
    "HR-MIN": hr_min,
    "PH": ph,
    "Absolute monocytes count": abs_mono,
    "Vent": vent_mapping[vent],
}])

# 确保列顺序与训练时一致
X_in = X_in[expected_features]

st.markdown("**Submitted Features:**")
st.dataframe(X_in)

# --- Prediction ---
st.subheader("Prediction")
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Run Prediction"):
    try:
        # 确保输入数据格式正确
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_in)
            # 对于二分类问题，取第二列的概率
            if proba.shape[1] > 1:
                proba = proba[:, 1]
            else:
                proba = proba[:, 0]
            
            proba_value = proba[0] if isinstance(proba, (np.ndarray, pd.Series)) else proba
            y_pred = int(proba_value >= threshold)

            st.success(f"Predicted probability of gut dysbiosis: {proba_value:.2%}")
            if y_pred == 1:
                st.warning("⚠ High Risk: Consider early gut function assessment, nutritional optimization, probiotics/prebiotics, and close monitoring.")
            else:
                st.info("✅ Low Risk: Continue standard management and reassess dynamically.")
        else:
            st.error("Model does not support probability predictions")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Input features:", X_in.dtypes)
        st.write("Expected feature types should match training data")