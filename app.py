import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Sepsis–ARDS Gut Dysbiosis Risk Predictor", layout="centered")

st.title("🧬 Sepsis–ARDS Gut Dysbiosis Risk Predictor")
st.caption("基于 Python/Streamlit 的在线预测工具。将您训练好的模型 final_model.pkl 放在同目录即可运行。")

# --- 1. 加载模型 ---
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
    st.error(f"未能加载模型（final_model.pkl）。\n错误信息：{load_err}")
    st.stop()

# --- 2. 特征输入区（可根据训练时的特征顺序与含义自行调整） ---
st.subheader("输入关键临床参数")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("年龄 (years)", min_value=18, max_value=100, value=60, step=1)
    spo2_max = st.number_input("SPO₂ 最大值 (%)", min_value=50, max_value=100, value=95, step=1)
    hr_min = st.number_input("心率最小值 HR-MIN (bpm)", min_value=20, max_value=200, value=60, step=1)

with col2:
    ph = st.number_input("动脉血气 pH", min_value=6.8, max_value=7.8, value=7.35, step=0.01, format="%.2f")
    abs_mono = st.number_input("绝对单核细胞计数 (×10⁹/L)", min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.2f")
    vent = st.selectbox("是否使用机械通气", ["否", "是"])

# 组装输入为 DataFrame，确保特征顺序与训练时一致
default_feature_names = ["age", "spo2_max", "ph", "absolute_monocytes", "vent", "hr_min"]
X_df = pd.DataFrame([{
    "age": age,
    "spo2_max": spo2_max,
    "ph": ph,
    "absolute_monocytes": abs_mono,
    "vent": 1 if vent == "是" else 0,
    "hr_min": hr_min,
}])

st.markdown("**当前提交的特征**：")
st.dataframe(X_df)

# --- 3. 预测参数设置 ---
st.subheader("预测设置")
threshold = st.slider("分类阈值（根据您的研究，建议 0.95）", 0.0, 1.0, 0.95, 0.01)

# --- 4. 执行预测 ---
pred_btn = st.button("开始预测")

if pred_btn:
    try:
        # 确保输入数据是DataFrame格式（符合scikit-learn模型的输入要求）
        prediction = model.predict(X_df)
        probability = model.predict_proba(X_df)[:, 1]  # 取阳性类别的概率
        
        # 计算基于阈值的分类结果
        y_pred = int(probability[0] >= threshold)
        
        st.success(f"预测概率（肠道菌群紊乱）: {probability[0]:.2%}")
        if y_pred == 1:
            st.warning("⚠ 高风险：建议尽早进行肠道功能评估、营养支持优化、益生菌/益生元等干预，并密切监测。")
        else:
            st.info("✅ 低风险：继续常规管理，并根据病情动态复评。")

        # --- 5. 模型可解释性部分 ---
        with st.expander("模型可解释性（提示：若需要 SHAP，请在模型训练时保留特征名或将 Pipeline 一并保存）"):
            # 尝试获取特征名
            feat_names = None
            if hasattr(model, "feature_names_in_"):
                feat_names = list(model.feature_names_in_)
            elif hasattr(model, "named_steps"):
                # 尝试从Pipeline中获取
                try:
                    last_step_name, last_step = list(model.named_steps.items())[-1]
                    if hasattr(last_step, "feature_names_in_"):
                        feat_names = list(last_step.feature_names_in_)
                except Exception:
                    pass

            if feat_names is None:
                feat_names = default_feature_names

            st.write("用于预测的特征顺序（请与训练保持一致）:", feat_names)
            st.write("该患者输入向量：")
            st.json(X_df.iloc[0].to_dict())
            st.caption("说明：若要显示 SHAP 值，请在训练阶段保留预处理管线（OneHot/StandardScaler等）与特征名，并在此处使用与训练一致的列。")

    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.stop()

st.divider()
st.markdown(
    "**机制背景（肠-肺轴）**：脓毒症导致的肠道屏障破坏与菌群紊乱可通过肠-肺轴加重肺部炎症，"
    "从而影响 ARDS 进程与预后。本工具旨在帮助医生更早识别高风险人群，争取早期干预窗口。"
)
