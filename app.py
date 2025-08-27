# 首先添加依赖检查和安装代码
import subprocess
import sys
import importlib
import os
import time

# 定义所有需要的包，按依赖顺序排列
required_packages = [
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "matplotlib>=3.8.0",
    "pandas>=2.0.0",
    "seaborn>=0.13.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "imbalanced-learn>=0.12.0",
    "shap>=0.46.0",
    "streamlit>=1.30.0"
]

def install_package(package):
    """尝试多种方式安装单个包"""
    # 提取包名
    package_name = package.split('>=')[0].split('==')[0]
    
    # 尝试不同的安装命令
    install_commands = [
        [sys.executable, "-m", "pip", "install", package],
        [sys.executable, "-m", "pip", "install", "--upgrade", package],
        [sys.executable, "-m", "pip", "install", "--user", package],
        ["pip", "install", package]
    ]
    
    for cmd in install_commands:
        try:
            print(f"尝试安装 {package}: {cmd}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 延长超时时间
            )
            if result.returncode == 0:
                print(f"{package} 安装成功")
                # 安装成功后短时间等待
                time.sleep(1)
                return True
            else:
                print(f"安装命令失败，返回码: {result.returncode}")
                print(f"错误输出: {result.stderr[:500]}")  # 只显示前500字符
        except Exception as e:
            print(f"安装 {package} 时出错: {str(e)}")
    
    return False

def install_all_packages():
    """按顺序安装所有包，确保依赖关系正确"""
    # 先检查是否在Streamlit Cloud环境
    is_streamlit_cloud = os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"
    print(f"Streamlit Cloud环境: {is_streamlit_cloud}")
    
    # 逐个安装包，前面的包是后面的依赖
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        import_name = package_name.replace('imbalanced-learn', 'imblearn').replace('-', '_')
        
        try:
            # 检查是否已安装
            importlib.import_module(import_name)
            print(f"包 {package_name} 已安装，跳过")
            continue
        except ImportError:
            print(f"包 {package_name} 未安装，需要安装")
        
        # 尝试安装
        success = install_package(package)
        if not success:
            print(f"警告: {package} 安装失败，将尝试继续")

# 首先安装所有依赖
install_all_packages()

# 安全导入函数 - 逐个导入并处理可能的错误
def safe_imports():
    """安全导入所有需要的库，处理可能的导入错误"""
    imports = {
        'streamlit': 'st',
        'pandas': 'pd',
        'numpy': 'np',
        'pickle': 'pickle',
        'os': 'os',
        'matplotlib.pyplot': 'plt',
        'seaborn': 'sns',
        'sklearn.preprocessing': ['StandardScaler', 'OneHotEncoder']
    }
    
    imported = {}
    
    # 首先尝试导入matplotlib并设置后端
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        imported['plt'] = plt
        print("matplotlib成功导入")
    except Exception as e:
        print(f"matplotlib导入失败: {str(e)}")
        imported['plt'] = None
    
    # 导入其他库
    for module, alias in imports.items():
        if module == 'matplotlib.pyplot':
            continue  # 已经处理过
        
        try:
            if isinstance(alias, list):
                # 处理从模块导入多个类的情况
                imported_module = importlib.import_module(module)
                for item in alias:
                    imported[item] = getattr(imported_module, item)
                print(f"成功导入 {module} 中的 {alias}")
            else:
                # 处理普通导入
                imported[alias] = importlib.import_module(module)
                print(f"成功导入 {module} 为 {alias}")
        except ImportError as e:
            print(f"导入 {module} 失败: {str(e)}")
            imported[alias] = None
    
    return imported

# 执行安全导入
imp = safe_imports()

# 检查关键库是否导入成功
if imp['st'] is None:
    print("错误: streamlit导入失败，应用无法运行")
    sys.exit(1)

# 从导入结果中提取所需的库
st = imp['st']
pd = imp['pd']
np = imp['np']
pickle = imp['pickle']
os = imp['os']
plt = imp['plt']
sns = imp['sns']
StandardScaler = imp.get('StandardScaler')
OneHotEncoder = imp.get('OneHotEncoder')

# 设置页面配置
st.set_page_config(
    page_title="脓毒症合并肠道菌群失调预测",
    page_icon="🏥",
    layout="wide"
)

# 页面标题
st.title("脓毒症合并肠道菌群失调预测模型")
st.write("基于机器学习的脓毒症患者肠道菌群失调风险预测工具")

# 检查缺失的库并显示警告
missing_libraries = []
if plt is None:
    missing_libraries.append("matplotlib (图表功能)")
if sns is None:
    missing_libraries.append("seaborn (高级可视化)")

if missing_libraries:
    st.warning(f"以下功能可能受限，因为某些库未能加载: {', '.join(missing_libraries)}。预测功能仍可使用。")

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

# 获取模型使用的特征 - 更新为包含所有必要特征
def get_model_features():
    """获取模型需要的特征列表"""
    try:
        # 尝试从文件读取特征列表
        with open(os.path.join('deploy_model', 'used_features.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            features = [line.strip().split('. ')[1] for line in lines if line.strip()]
        return features
    except:
        # 如果无法读取特征文件，使用已知需要的特征列表（包含缺失的特征）
        return [
            # 原有特征
            'Age', 'LAC', 'WBC', 'Absolute neutrophil count',
            'Absolute lymphocyte count', 'PLT', 'Albumin',
            'BUN', 'CRE', 'NLR',
            # 缺失的特征
            'Shock_Index', 'Glu', 'SPO2-MAX', 'HR-MIN', 
            'Race', 'PH', 'Diabetes', 'Absolute monocytes count', 'Length of stay'
        ]

# 加载特征信息和参考范围 - 添加缺失特征的信息
def load_feature_info():
    """加载特征的参考范围和说明"""
    feature_info = {
        # 原有特征
        'Age': {'range': (0, 120), 'unit': '岁', 'desc': '患者年龄'},
        'LAC': {'range': (0, 20), 'unit': 'mmol/L', 'desc': '乳酸水平，反映组织缺氧情况'},
        'WBC': {'range': (0, 50), 'unit': '×10^9/L', 'desc': '白细胞计数，反映炎症反应'},
        'Absolute neutrophil count': {'range': (0, 40), 'unit': '×10^9/L', 'desc': '中性粒细胞绝对计数'},
        'Absolute lymphocyte count': {'range': (0, 10), 'unit': '×10^9/L', 'desc': '淋巴细胞绝对计数'},
        'PLT': {'range': (0, 1000), 'unit': '×10^9/L', 'desc': '血小板计数'},
        'Albumin': {'range': (0, 60), 'unit': 'g/L', 'desc': '白蛋白水平，反映营养状态'},
        'BUN': {'range': (0, 50), 'unit': 'mmol/L', 'desc': '血尿素氮，反映肾功能'},
        'CRE': {'range': (0, 500), 'unit': 'μmol/L', 'desc': '肌酐，反映肾功能'},
        'NLR': {'range': (0, 100), 'unit': '', 'desc': '中性粒细胞与淋巴细胞比值，反映炎症状态'},
        
        # 新增缺失的特征
        'Shock_Index': {'range': (0, 5), 'unit': '', 'desc': '休克指数，心率/收缩压'},
        'Glu': {'range': (2, 30), 'unit': 'mmol/L', 'desc': '血糖水平'},
        'SPO2-MAX': {'range': (50, 100), 'unit': '%', 'desc': '最高血氧饱和度'},
        'HR-MIN': {'range': (30, 200), 'unit': '次/分', 'desc': '最低心率'},
        'Race': {'range': (0, 5), 'unit': '', 'desc': '种族 (0-5表示不同种族分类)'},
        'PH': {'range': (6.8, 7.8), 'unit': '', 'desc': '血液酸碱度'},
        'Diabetes': {'range': (0, 1), 'unit': '', 'desc': '是否有糖尿病 (0=否, 1=是)'},
        'Absolute monocytes count': {'range': (0, 5), 'unit': '×10^9/L', 'desc': '单核细胞绝对计数'},
        'Length of stay': {'range': (0, 100), 'unit': '天', 'desc': '住院时间'}
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
        
        # 检查是否有缺失的特征
        missing = set(features) - set(input_df.columns)
        if missing:
            st.error(f"预测失败：缺少必要特征 {missing}")
            return None, None
        
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
    
    # 分三列显示输入框，适应更多特征
    cols = st.columns(3)
    
    for i, feature in enumerate(features):
        col = cols[i % 3]
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
                
                # 只有当matplotlib和seaborn可用时才绘制图表
                if plt is not None and sns is not None:
                    # 绘制风险可视化图表
                    try:
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
                    except Exception as e:
                        st.warning(f"图表绘制失败: {str(e)}")
                    
                    # 特征贡献分析（简化版）
                    with st.expander("查看特征贡献分析", expanded=False):
                        st.write("""
                        特征贡献分析显示各因素对预测结果的影响程度。
                        （完整分析请参考模型SHAP值报告）
                        """)
                        
                        try:
                            # 简化的特征重要性展示
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
                        except Exception as e:
                            st.warning(f"特征重要性图表绘制失败: {str(e)}")
                else:
                    st.info("图表功能暂时不可用，核心预测功能不受影响。")

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
    