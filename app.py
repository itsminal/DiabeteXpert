import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import io
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings


# Set page configuration
st.set_page_config(page_title="DiabeteXpert", layout="wide")

# App title and description
st.title("Predicting Wellness, One Step at a Time")
st.markdown("""
A machine learning-powered system that analyzes health data to predict diabetes risk, enabling early intervention and proactive health management.
""")

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    
    # Check if directory exists
    if not os.path.exists("saved_models"):
        st.warning("'saved_models' directory not found. Creating directory...")
        os.makedirs("saved_models", exist_ok=True)
        return models
    
    # Try multiple approaches to load models
    model_files = ["all_models_joblib.pkl", "all_models_pickle.pkl", "all_models.pkl", "models.pkl"]
    
    # Try loading from a single file first
    for filename in model_files:
        filepath = os.path.join("saved_models", filename)
        if os.path.exists(filepath):
            try:
                # Try joblib first
                try:
                    models = joblib.load(filepath)
                    st.success(f"Successfully loaded models from {filename}")
                    return models
                except:
                    # Try pickle if joblib fails
                    with open(filepath, 'rb') as f:
                        models = pickle.load(f)
                    st.success(f"Successfully loaded models from {filename} using pickle")
                    return models
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")
    
    # If we couldn't load all models at once, try individual files
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for name in model_names:
        # Try standard filename
        filepath = os.path.join("saved_models", f"{name}.pkl")
        if os.path.exists(filepath):
            try:
                models[name] = joblib.load(filepath)
                st.success(f"Loaded {name} model")
            except:
                st.warning(f"Failed to load {name} model")
                
        # Try alternative filename format
        clean_name = name.replace(' ', '_')
        filepath = os.path.join("saved_models", f"{clean_name}.pkl")
        if os.path.exists(filepath):
            try:
                models[name] = joblib.load(filepath)
                st.success(f"Loaded {name} model (using {clean_name}.pkl)")
            except:
                st.warning(f"Failed to load {name} model (using {clean_name}.pkl)")
    
    if not models:
        st.error("No models could be loaded. Please check your saved_models directory.")
        
    return models

# Function to create test models if needed
def create_test_models():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        # Create simple test models
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        test_models = {
            'Logistic Regression': LogisticRegression().fit(X, y),
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y),
            'XGBoost': XGBClassifier(n_estimators=10, random_state=42).fit(X, y)
        }
        
        # Save directory
        os.makedirs("saved_models", exist_ok=True)
        
        # Save with joblib
        joblib.dump(test_models, 'saved_models/all_models_joblib.pkl')
        
        st.success("Test models created and saved successfully!")
        return test_models
        
    except Exception as e:
        st.error(f"Error creating test models: {e}")
        return {}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Manual Prediction", "Batch Prediction", "Model Info", "Debug"])

# Load models
models = load_models()
model_names = list(models.keys()) if models else []

# If no models are found, show an error
if not model_names and page != "Debug":
    st.error("No models found. Go to Debug page to create test models or check your model files.")

# Manual Prediction page
if page == "Manual Prediction":
    st.header("Manual Feature Input for Prediction")
    
    if not model_names:
        st.warning("No models available. Please go to the Debug page.")
    else:
        # Select model for prediction
        selected_model = st.selectbox("Select model for prediction:", model_names)
        model = models[selected_model]
        
        # Get feature names
        # Try different approaches to get feature names
        feature_names = []
        
        # Method 1: Check if model has feature_names_ attribute
        if hasattr(model, 'feature_names_'):
            feature_names = model.feature_names_
        # Method 2: For some sklearn models
        elif hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        
        # If we couldn't extract feature names, ask the user
        if feature_names is None:
            st.info("Could not automatically determine feature names. Please enter them manually.")
            features_input = st.text_input("Enter feature names separated by commas:", "feature1, feature2, feature3")
            feature_names = [name.strip() for name in features_input.split(',')]
        
        # Display feature input fields
        st.subheader("Enter Feature Values")
        
        # Create columns for input fields
        feature_values = {}
        
        # Create 2 columns for input
        num_features = len(feature_names)
        col1, col2 = st.columns(2)
        
        # Distribute features across the columns
        for i, feature in enumerate(feature_names):
            with col1 if i < num_features/2 else col2:
                if feature == 'gender':
                    op = st.selectbox(f"{feature}", options=["Female", "Male", "Other"])
                    feature_values[feature] = ["Female", "Male", "Other"].index(op)

                elif feature == 'smoking_history':
                    op = st.selectbox(f"{feature}", options=['No Info', 'current', 'ever', 'former', 'never', 'not current'])
                    feature_values[feature] = ['No Info', 'current', 'ever', 'former', 'never', 'not current'].index(op)
                else:
                    feature_values[feature] = st.number_input(f"{feature}:", format="%.4f", step=0.1)
        
        
        # Make prediction
        if st.button("Predict"):
            try:
                # Create input data from entered values
                input_data = pd.DataFrame([feature_values])
                scaler = joblib.load('scaler/std_scaler.pkl')
                input_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(
                    input_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
                # Display the input data
                st.subheader("Input Data")
                st.write(input_data)
                
                # Make prediction
                prediction = model.predict(input_data)
                
                st.subheader("Prediction Result")
                st.write(f"**Predicted Class:** {prediction[0]}")
                
                # Try to get prediction probability if available
                try:
                    proba = model.predict_proba(input_data)
                    st.write("**Class Probabilities:**")
                    
                    # Create a dataframe for probabilities
                    proba_df = pd.DataFrame(
                        proba[0], 
                        index=model.classes_ if hasattr(model, 'classes_') else range(len(proba[0])),
                        columns=['Probability']
                    )
                    
                    st.write(proba_df)
                    
                    # Create a bar chart for probabilities
                    fig, ax = plt.subplots()
                    proba_df.plot.bar(ax=ax)
                    ax.set_title('Class Probabilities')
                    ax.set_ylabel('Probability')
                    ax.set_xlabel('Class')
                    st.pyplot(fig)
                    
                except:
                    st.info("Probability prediction not available for this model.")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Batch Prediction page
elif page == "Batch Prediction":
    st.header("Batch Prediction from CSV")
    
    if not model_names:
        st.warning("No models available. Please go to the Debug page.")
    else:
        uploaded_file = st.file_uploader("Choose a CSV file with feature data:", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the data
                data = pd.read_csv(uploaded_file)
                
                # Display data preview
                st.subheader("Data Preview")
                st.write(data.head())
                
                # Select models for prediction
                st.subheader("Select Models")
                selected_models = st.multiselect("Choose models to use:", model_names, default=model_names[0])
                
                if selected_models and st.button("Generate Predictions"):
                    try:
                        # Create a DataFrame to store predictions
                        results = pd.DataFrame()
                        
                        # Make predictions with selected models
                        for model_name in selected_models:
                            model = models[model_name]
                            
                            # Binary class predictions
                            results[f"{model_name}_Prediction"] = model.predict(data)
                            
                            # Get probability predictions if available
                            try:
                                probs = model.predict_proba(data)
                                # If binary classification, take probability of class 1
                                if probs.shape[1] == 2:
                                    results[f"{model_name}_Probability"] = probs[:, 1]
                                else:
                                    # For multiclass, just note the max probability
                                    results[f"{model_name}_Max_Probability"] = np.max(probs, axis=1)
                            except:
                                pass
                        
                        # Create ensemble prediction if multiple models selected
                        if len(selected_models) > 1:
                            pred_columns = [col for col in results.columns if 'Prediction' in col]
                            results['Ensemble_Vote'] = results[pred_columns].mode(axis=1)[0]
                            
                            # Try to get ensemble probability if probability columns exist
                            prob_columns = [col for col in results.columns if 'Probability' in col]
                            if prob_columns:
                                results['Ensemble_Probability'] = results[prob_columns].mean(axis=1)
                        
                        # Display predictions
                        st.subheader("Prediction Results")
                        st.write(results.head(20))
                        
                        # Combine original data with predictions
                        combined = pd.concat([data, results], axis=1)
                        
                        # Download predictions
                        csv = combined.to_csv(index=False)
                        st.download_button(
                            label="Download All Results as CSV",
                            data=csv,
                            file_name="model_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {e}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# Model Info page
elif page == "Model Info":
    st.header("Model Information")
    
    if not model_names:
        st.warning("No models available. Please go to the Debug page.")
    else:
        # Select model to view info
        selected_model = st.selectbox("Select model:", model_names)
        model = models[selected_model]
        
        # Display model details
        st.subheader(f"{selected_model} Model Details")
        
        # Model type
        st.write(f"**Model Type:** {type(model).__name__}")
        
        # Model parameters
        st.write("**Model Parameters:**")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
            st.write(params_df)
        else:
            st.write("Parameter information not available")
        
        # Feature importance for applicable models
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            st.subheader("Feature Importance")
            
            # Get feature names
            feature_names = []
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            elif 'session_state' in st.__dict__ and 'X' in st.session_state:
                feature_names = st.session_state['X'].columns
            
            if feature_names is None:
                st.info("Feature names not available. Using generic names.")
                if hasattr(model, 'feature_importances_'):
                    feature_names = [f"Feature {i+1}" for i in range(len(model.feature_importances_))]
                elif hasattr(model, 'coef_') and len(model.coef_.shape) > 1:
                    feature_names = [f"Feature {i+1}" for i in range(model.coef_.shape[1])]
                else:
                    feature_names = [f"Feature {i+1}" for i in range(10)]  # Default placeholder
            
            # Different models store feature importance differently
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                importances = None
            
            if importances is not None and len(feature_names) == len(importances):
                # Create DataFrame for feature importance
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax)
                ax.set_title(f'Top 15 Feature Importance - {selected_model}')
                st.pyplot(fig)
                
                # Display table of all feature importances
                st.write("Feature Importance Table:")
                st.write(importance_df)
            else:
                st.info("Feature importance visualization not available")

# Debug page
elif page == "Debug":
    st.header("Debug and Troubleshooting")
    
    st.subheader("Check Saved Models Directory")
    # Check if directory exists
    if os.path.exists("saved_models"):
        files = os.listdir("saved_models")
        st.write("Files in saved_models directory:")
        for file in files:
            st.write(f"- {file}")
    else:
        st.error("The 'saved_models' directory does not exist!")
    
    st.subheader("Create Test Models")
    if st.button("Create Sample Models for Testing"):
        models = create_test_models()
        if models:
            st.success("Test models created successfully! Please restart the app to use them.")
    
    st.subheader("Try Manual Model Loading")
    model_path = st.text_input("Enter the full path to a model file:", "saved_models/all_models_joblib.pkl")
    
    if st.button("Test Load Model"):
        try:
            # Try joblib first
            try:
                model = joblib.load(model_path)
                st.success("Successfully loaded using joblib")
                st.write("Model type:", type(model))
                if isinstance(model, dict):
                    st.write("Dictionary keys:", list(model.keys()))
            except Exception as e_joblib:
                st.warning(f"Joblib loading failed: {e_joblib}")
                
                # Try pickle if joblib fails
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    st.success("Successfully loaded using pickle")
                    st.write("Model type:", type(model))
                    if isinstance(model, dict):
                        st.write("Dictionary keys:", list(model.keys()))
                except Exception as e_pickle:
                    st.error(f"Pickle loading also failed: {e_pickle}")
                    
        except Exception as e:
            st.error(f"Error in loading process: {e}")
    
    st.subheader("Environment Information")
    import sys
    st.write(f"Python version: {sys.version}")
    st.write(f"Joblib version: {joblib.__version__}")
    
    try:
        import sklearn
        st.write(f"Scikit-learn version: {sklearn.__version__}")
    except:
        st.warning("Scikit-learn not imported")
    
    try:
        import xgboost
        st.write(f"XGBoost version: {xgboost.__version__}")
    except:
        st.warning("XGBoost not imported")