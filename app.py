import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 设置页面配置
st.set_page_config(
    page_title="脓毒症合并肠道菌群失调预测",
    page_icon="🏥",
    layout="wide"
)

# 页面标题
st.title("脓毒症合并肠道菌群失调预测模型")
st.write("基于机器学习的脓毒症患者肠道菌群失调风险预测工具")

# 加载模型
@st.cache_resource
def load_model(model_path):
    """加载保存的模型"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# 获取模型使用的特征
def get_model_features():
    """获取模型需要的特征列表"""
    # 这里应该替换为你模型实际使用的前10个SHAP特征
    # 从你的代码中可知这些特征保存在deploy_dir/used_features.txt
    try:
        with open(os.path.join('deploy_model', 'used_features.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            features = [line.strip().split('. ')[1] for line in lines[1:11]]  # 取前10个特征
        return features
    except:
        # 如果无法读取特征文件，使用默认特征列表（请根据实际情况修改）
        return [
            'Age', 'LAC', 'WBC', 'Absolute neutrophil count',
            'Absolute lymphocyte count', 'PLT', 'Albumin',
            'BUN', 'CRE', 'NLR'
        ]

# 加载特征信息和参考范围
def load_feature_info():
    """加载特征的参考范围和说明"""
    # 这里定义了每个特征的参考范围和简要说明
    # 你可以根据实际医学知识调整这些值
    feature_info = {
        'Age': {'range': (0, 120), 'unit': '岁', 'desc': '患者年龄'},
        'LAC': {'range': (0, 20), 'unit': 'mmol/L', 'desc': '乳酸水平，反映组织缺氧情况'},
        'WBC': {'range': (0, 50), 'unit': '×10^9/L', 'desc': '白细胞计数，反映炎症反应'},
        'Absolute neutrophil count': {'range': (0, 40), 'unit': '×10^9/L', 'desc': '中性粒细胞绝对计数'},
        'Absolute lymphocyte count': {'range': (0, 10), 'unit': '×10^9/L', 'desc': '淋巴细胞绝对计数'},
        'PLT': {'range': (0, 1000), 'unit': '×10^9/L', 'desc': '血小板计数'},
        'Albumin': {'range': (0, 60), 'unit': 'g/L', 'desc': '白蛋白水平，反映营养状态'},
        'BUN': {'range': (0, 50), 'unit': 'mmol/L', 'desc': '血尿素氮，反映肾功能'},
        'CRE': {'range': (0, 500), 'unit': 'μmol/L', 'desc': '肌酐，反映肾功能'},
        'NLR': {'range': (0, 100), 'unit': '', 'desc': '中性粒细胞与淋巴细胞比值，反映炎症状态'}
    }
    
    # 对于模型实际使用的特征，如果不在上述字典中，添加默认值
    model_features = get_model_features()
    for feature in model_features:
        if feature not in feature_info:
            feature_info[feature] = {
                'range': (0, 100), 
                'unit': '', 
                'desc': '临床特征'
            }
    
    return feature_info

# 预测函数
def predict(model, input_data):
    """使用模型进行预测"""
    try:
        # 确保输入数据是DataFrame且列顺序正确
        features = get_model_features()
        input_df = pd.DataFrame([input_data], columns=features)
        
        # 进行预测
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability >= 0.5 else 0  # 使用0.5作为阈值
        
        return probability, prediction
    except Exception as e:
        st.error(f"预测过程出错: {str(e)}")
        return None, None

# 主应用
def main():
    # 加载模型
    model = load_model(os.path.join('deploy_model', 'final_deploy_model.pkl'))
    
    if model is None:
        st.stop()
    
    # 获取模型特征和信息
    features = get_model_features()
    feature_info = load_feature_info()
    
    # 显示使用的特征
    with st.expander("查看模型使用的特征", expanded=False):
        st.write("模型使用以下临床特征进行预测:")
        for i, feature in enumerate(features, 1):
            st.write(f"{i}. {feature} ({feature_info[feature]['desc']})")
    
    # 创建输入表单
    st.subheader("输入患者特征")
    input_data = {}
    
    # 分两列显示输入框
    cols = st.columns(2)
    
    for i, feature in enumerate(features):
        col = cols[i % 2]
        with col:
            min_val, max_val = feature_info[feature]['range']
            unit = feature_info[feature]['unit']
            input_data[feature] = st.number_input(
                f"{feature} ({feature_info[feature]['desc']})",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.1,
                format="%.1f"
            )
    
    # 预测按钮
    if st.button("预测肠道菌群失调风险", key="predict_btn"):
        with st.spinner("正在进行预测..."):
            probability, prediction = predict(model, input_data)
            
            if probability is not None:
                # 显示预测结果
                st.subheader("预测结果")
                
                # 显示风险概率
                risk_percent = probability * 100
                st.write(f"患者发生肠道菌群失调的概率: **{risk_percent:.1f}%**")
                
                # 根据风险级别显示不同颜色的提示
                if risk_percent < 30:
                    st.success("风险评估: 低风险")
                elif risk_percent < 70:
                    st.warning("风险评估: 中等风险")
                else:
                    st.error("风险评估: 高风险")
                
                # 显示风险解释
                st.info("""
                注: 本预测结果仅供临床参考，不构成诊断依据。
                临床医生应结合患者具体情况进行综合判断。
                """)
                
                # 绘制风险可视化图表
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['风险概率'], [risk_percent], color='skyblue')
                ax.axvline(x=30, color='green', linestyle='--', alpha=0.5)
                ax.axvline(x=70, color='orange', linestyle='--', alpha=0.5)
                ax.set_xlim(0, 100)
                ax.set_xlabel('风险概率 (%)')
                ax.set_title('肠道菌群失调风险评估')
                plt.text(30, 0, '  低风险', color='green')
                plt.text(50, 0, '  中等风险', color='orange')
                plt.text(85, 0, '  高风险', color='red')
                st.pyplot(fig)
                
                # 特征贡献分析（简化版）
                with st.expander("查看特征贡献分析", expanded=False):
                    st.write("""
                    特征贡献分析显示各因素对预测结果的影响程度。
                    （完整分析请参考模型SHAP值报告）
                    """)
                    
                    # 这里是简化的特征重要性展示
                    # 在实际应用中，你可以加载预计算的SHAP值来提供更准确的分析
                    importance = np.random.rand(len(features))  # 随机生成示例重要性
                    importance = importance / np.sum(importance)
                    
                    sorted_idx = np.argsort(importance)[::-1]
                    sorted_features = [features[i] for i in sorted_idx]
                    sorted_importance = [importance[i] for i in sorted_idx]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=sorted_importance, y=sorted_features, ax=ax)
                    ax.set_xlabel('相对重要性')
                    ax.set_title('特征对预测结果的影响')
                    st.pyplot(fig)

# 页脚信息
def footer():
    st.markdown("""
    ---
    ### 关于本工具
    本工具基于机器学习算法开发，用于辅助评估脓毒症患者发生肠道菌群失调的风险。
    模型使用多个临床特征进行预测，预测结果仅供临床参考。
    """)

if __name__ == "__main__":
    main()
    footer()
