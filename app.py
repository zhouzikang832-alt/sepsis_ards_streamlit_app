import subprocess
import sys
import importlib
import os
import time

# Define all required packages in dependency order
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
    """Attempt to install a single package using multiple methods"""
    # Extract package name
    package_name = package.split('>=')[0].split('==')[0]
    
    # Try different installation commands
    install_commands = [
        [sys.executable, "-m", "pip", "install", package],
        [sys.executable, "-m", "pip", "install", "--upgrade", package],
        [sys.executable, "-m", "pip", "install", "--user", package],
        ["pip", "install", package]
    ]
    
    for cmd in install_commands:
        try:
            print(f"Attempting to install {package}: {cmd}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # Extend timeout period
            )
            if result.returncode == 0:
                print(f"{package} installed successfully")
                # Short wait after successful installation
                time.sleep(1)
                return True
            else:
                print(f"Installation command failed with return code: {result.returncode}")
                print(f"Error output: {result.stderr[:500]}")  # Show only first 500 characters
        except Exception as e:
            print(f"Error installing {package}: {str(e)}")
    
    return False

def install_all_packages():
    """Install all packages in sequence to ensure correct dependencies"""
    # Check if running in Streamlit Cloud environment
    is_streamlit_cloud = os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"
    print(f"Streamlit Cloud environment: {is_streamlit_cloud}")
    
    # Install packages one by one, with earlier packages being dependencies for later ones
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        import_name = package_name.replace('imbalanced-learn', 'imblearn').replace('-', '_')
        
        try:
            # Check if already installed
            importlib.import_module(import_name)
            print(f"Package {package_name} is already installed, skipping")
            continue
        except ImportError:
            print(f"Package {package_name} is not installed, needs installation")
        
        # Attempt installation
        success = install_package(package)
        if not success:
            print(f"Warning: {package} installation failed, will attempt to continue")

# First install all dependencies
install_all_packages()

# Safe import function - import one by one and handle potential errors
def safe_imports():
    """Safely import all required libraries, handling potential import errors"""
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
    
    # First attempt to import matplotlib and set backend
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        imported['plt'] = plt
        print("matplotlib imported successfully")
    except Exception as e:
        print(f"matplotlib import failed: {str(e)}")
        imported['plt'] = None
    
    # Import other libraries
    for module, alias in imports.items():
        if module == 'matplotlib.pyplot':
            continue  # Already handled
        
        try:
            if isinstance(alias, list):
                # Handle importing multiple classes from a module
                imported_module = importlib.import_module(module)
                for item in alias:
                    imported[item] = getattr(imported_module, item)
                print(f"Successfully imported {alias} from {module}")
            else:
                # Handle regular imports
                imported[alias] = importlib.import_module(module)
                print(f"Successfully imported {module} as {alias}")
        except ImportError as e:
            print(f"Failed to import {module}: {str(e)}")
            imported[alias] = None
    
    return imported

# Execute safe imports
imp = safe_imports()

# Check if critical libraries were imported successfully
if imp['st'] is None:
    print("Error: streamlit import failed, application cannot run")
    sys.exit(1)

# Extract required libraries from import results
st = imp['st']
pd = imp['pd']
np = imp['np']
pickle = imp['pickle']
os = imp['os']
plt = imp['plt']
sns = imp['sns']
StandardScaler = imp.get('StandardScaler')
OneHotEncoder = imp.get('OneHotEncoder')

# Set page configuration
st.set_page_config(
    page_title="Sepsis-Associated Gut Dysbiosis Prediction",
    page_icon="🏥",
    layout="wide"
)

# Page title
st.title("Sepsis-Associated Gut Dysbiosis Prediction Model")
st.write("Machine learning-based tool for predicting gut dysbiosis risk in sepsis patients")

# Check for missing libraries and display warnings
missing_libraries = []
if plt is None:
    missing_libraries.append("matplotlib (chart functionality)")
if sns is None:
    missing_libraries.append("seaborn (advanced visualization)")

if missing_libraries:
    st.warning(f"The following features may be limited because some libraries failed to load: {', '.join(missing_libraries)}. Prediction functionality should still be available.")

# Load model
@st.cache_resource
def load_model(model_path):
    """Load the saved model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Get features used by the model - updated to include only the 10 specified features
def get_model_features():
    """Get the list of features required by the model"""
    return [
        'Absolute monocytes count',
        'SPO2-MAX',
        'Vent',
        'RR-MIN',
        'DBP-MAX',
        'PH',
        'Glu',
        'Race',
        'INR',
        'Lactate_Albumin_Ratio'
    ]

# Define race categories mapping
race_categories = {
    "ASIAN": 0,
    "WHITE": 1,
    "BLACK": 2,
    "HISPANIC": 3,
    "OTHER": 4
}

# Load feature information and reference ranges
def load_feature_info():
    """Load feature reference ranges and descriptions"""
    feature_info = {
        'Absolute monocytes count': {
            'range': (0, 5), 
            'unit': '×10^9/L', 
            'desc': 'Absolute monocyte count in blood'
        },
        'SPO2-MAX': {
            'range': (50, 100), 
            'unit': '%', 
            'desc': 'Maximum oxygen saturation'
        },
        'Vent': {
            'range': (0, 1), 
            'unit': '', 
            'desc': 'Ventilation status (0=Not ventilated, 1=Ventilated)'
        },
        'RR-MIN': {
            'range': (5, 60), 
            'unit': 'breaths/min', 
            'desc': 'Minimum respiratory rate'
        },
        'DBP-MAX': {
            'range': (30, 150), 
            'unit': 'mmHg', 
            'desc': 'Maximum diastolic blood pressure'
        },
        'PH': {
            'range': (6.8, 7.8), 
            'unit': '', 
            'desc': 'Blood pH level'
        },
        'Glu': {
            'range': (2, 30), 
            'unit': 'mmol/L', 
            'desc': 'Blood glucose level'
        },
        'Race': {
            'range': (0, 4), 
            'unit': '', 
            'desc': f'Race (categorical: {", ".join([f"{k}={v}" for k, v in race_categories.items()])})'
        },
        'INR': {
            'range': (0.5, 10), 
            'unit': '', 
            'desc': 'International normalized ratio (coagulation marker)'
        },
        'Lactate_Albumin_Ratio': {
            'range': (0, 20), 
            'unit': '', 
            'desc': 'Ratio of lactate to albumin'
        }
    }
    
    return feature_info

# Get SHAP values for feature importance
def get_shap_values():
    """Return the SHAP values for each feature"""
    return {
        'Absolute monocytes count': 1.6864873532313893,
        'SPO2-MAX': 0.9108963211559987,
        'Vent': 0.8160659120425647,
        'RR-MIN': 0.3788268790715488,
        'DBP-MAX': 0.37830679075447693,
        'PH': 0.32379738754271825,
        'Glu': 0.30925520532585865,
        'Race': 0.2655423627800955,
        'INR': 0.2371198053945889,
        'Lactate_Albumin_Ratio': 0.23178658952214662
    }

# Prediction function
def predict(model, input_data):
    """Make predictions using the model"""
    try:
        # Ensure input data is a DataFrame with correct column order
        features = get_model_features()
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Check for missing features
        missing = set(features) - set(input_df.columns)
        if missing:
            st.error(f"Prediction failed: missing required features {missing}")
            return None, None
        
        # Make prediction
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability >= 0.5 else 0  # Using 0.5 as threshold
        
        return probability, prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Main application
def main():
    # Load model
    model = load_model(os.path.join('deploy_model', 'final_deploy_model.pkl'))
    
    if model is None:
        st.stop()
    
    # Get model features and information
    features = get_model_features()
    feature_info = load_feature_info()
    shap_values = get_shap_values()
    
    # Display features used with SHAP values
    with st.expander("View features used by the model with SHAP values", expanded=False):
        st.write("The model uses the following clinical features for prediction, sorted by importance:")
        # Sort features by SHAP value descending
        sorted_features = sorted(features, key=lambda x: shap_values[x], reverse=True)
        
        for i, feature in enumerate(sorted_features, 1):
            st.write(f"{i}. {feature} - Mean Absolute SHAP Value: {shap_values[feature]:.4f}")
            st.write(f"   Description: {feature_info[feature]['desc']}")
    
    # Create input form
    st.subheader("Enter Patient Features")
    input_data = {}
    
    # Display input fields in two columns
    cols = st.columns(2)
    
    for i, feature in enumerate(features):
        col = cols[i % 2]
        with col:
            min_val, max_val = feature_info[feature]['range']
            unit = feature_info[feature]['unit']
            
            # Special handling for Race (dropdown selection)
            if feature == 'Race':
                race_option = st.selectbox(
                    f"{feature} ({feature_info[feature]['desc']})",
                    options=list(race_categories.keys())
                )
                input_data[feature] = race_categories[race_option]
            
            # Special handling for Vent (only 0 or 1)
            elif feature == 'Vent':
                vent_option = st.radio(
                    f"{feature} ({feature_info[feature]['desc']})",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )
                input_data[feature] = vent_option
            
            # Regular numeric input for other features
            else:
                input_data[feature] = st.number_input(
                    f"{feature} ({feature_info[feature]['desc']})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    format="%.1f"
                )
    
    # Prediction button
    if st.button("Predict Gut Dysbiosis Risk", key="predict_btn"):
        with st.spinner("Making prediction..."):
            probability, prediction = predict(model, input_data)
            
            if probability is not None:
                # Display prediction results
                st.subheader("Prediction Result")
                
                # Display risk probability
                risk_percent = probability * 100
                st.write(f"Probability of developing gut dysbiosis: **{risk_percent:.1f}%**")
                
                # Display risk assessment with color coding
                if risk_percent < 30:
                    st.success("Risk Assessment: Low Risk")
                elif risk_percent < 70:
                    st.warning("Risk Assessment: Moderate Risk")
                else:
                    st.error("Risk Assessment: High Risk")
                
                # Display risk explanation
                st.info("""
                Note: This prediction result is for clinical reference only and does not constitute a diagnosis.
                Clinicians should make comprehensive judgments based on the patient's specific situation.
                """)
                
                # Only plot charts if matplotlib and seaborn are available
                if plt is not None and sns is not None:
                    # Plot risk visualization chart
                    try:
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh(['Risk Probability'], [risk_percent], color='skyblue')
                        ax.axvline(x=30, color='green', linestyle='--', alpha=0.5)
                        ax.axvline(x=70, color='orange', linestyle='--', alpha=0.5)
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Risk Probability (%)')
                        ax.set_title('Gut Dysbiosis Risk Assessment')
                        plt.text(30, 0, '  Low Risk', color='green')
                        plt.text(50, 0, '  Moderate Risk', color='orange')
                        plt.text(85, 0, '  High Risk', color='red')
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Chart rendering failed: {str(e)}")
                    
                    # Feature contribution analysis using SHAP values
                    with st.expander("View Feature Contribution Analysis", expanded=False):
                        st.write("""
                        Feature contribution analysis shows the degree of influence of each factor on the prediction result,
                        based on Mean Absolute SHAP Values.
                        """)
                        
                        try:
                            # Get and sort SHAP values
                            features_sorted = sorted(shap_values.keys(), key=lambda x: shap_values[x], reverse=True)
                            importance = [shap_values[feature] for feature in features_sorted]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x=importance, y=features_sorted, ax=ax, palette='viridis')
                            ax.set_xlabel('Mean Absolute SHAP Value')
                            ax.set_title('Feature Importance Based on SHAP Values')
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Feature importance chart rendering failed: {str(e)}")
                else:
                    st.info("Chart functionality is temporarily unavailable, but core prediction functionality is unaffected.")

# Footer information
def footer():
    st.markdown("""
    ---
    ### About This Tool
    This tool is developed based on machine learning algorithms to assist in assessing the risk of gut dysbiosis in sepsis patients.
    The model uses multiple clinical features for prediction, and the prediction results are for clinical reference only.
    """)

if __name__ == "__main__":
    main()
    footer()
