# DiabeteXpert: Predicting Wellness, One Step at a Time

## Live Application
**Try the application**: [DiabeteXpert - Streamlit App](https://itsminal-diabetexpert-app-osm094.streamlit.app/)

## Overview
DiabeteXpert is a machine learning-powered web application that analyzes health data to predict diabetes risk, enabling early intervention and proactive health management. The application uses multiple sophisticated machine learning models to provide accurate predictions and insights based on health metrics.

## Features

### 1. Multiple Prediction Models
- **Logistic Regression**: Provides baseline predictions with good interpretability
- **Random Forest**: Offers robust performance by combining multiple decision trees
- **XGBoost**: Delivers high accuracy through gradient boosting techniques
- **Ensemble Model**: Combines all three models through voting for improved accuracy and reliability

### 2. Multiple Prediction Methods
- **Manual Prediction**: Input individual health metrics for personalized risk assessment
- **Batch Prediction**: Upload CSV files with multiple patients' data for bulk analysis

### 3. Comprehensive Model Information
- Detailed model parameters
- Feature importance visualization
- Performance metrics

### 4. Ensemble Voting System
The ensemble model leverages the collective intelligence of all three models:
- Uses soft voting (probability-based) for more nuanced predictions
- Displays individual model predictions alongside the ensemble result
- Visualizes the voting process and confidence levels

## Health Metrics Used
DiabeteXpert analyzes the following health indicators:
- Age
- Gender
- BMI (Body Mass Index)
- HbA1c level (glycated hemoglobin)
- Blood glucose level
- Hypertension status
- Heart disease history
- Smoking history

## How to Use

### Manual Prediction
1. Navigate to the "Manual Prediction" page
2. Select a prediction model (individual model or ensemble)
3. Enter the patient's health metrics
4. Click "Predict" to see the risk assessment
5. Review the prediction, probability scores, and (if using ensemble) individual model predictions

### Batch Prediction
1. Navigate to the "Batch Prediction" page
2. Upload a CSV file containing multiple patients' data
3. Select one or more models for prediction
4. Click "Generate Predictions" to analyze all patients
5. Download the results as a CSV file for further analysis

## Technical Information
DiabeteXpert is built with:
- **Streamlit**: For the interactive web interface
- **Scikit-learn**: For Logistic Regression, Random Forest, and ensemble models
- **XGBoost**: For gradient boosting implementation
- **Pandas**: For data manipulation
- **Matplotlib & Seaborn**: For data visualization
- **Joblib**: For model persistence

## Benefits for Healthcare Providers
- Early identification of high-risk patients
- Evidence-based decision support
- Efficient screening of large patient populations
- Comparison of multiple prediction models for confidence assessment

## Benefits for Patients
- Early awareness of potential diabetes risk
- Opportunity for preventive interventions
- Educational insights on key risk factors
- Personalized risk assessment

## Debug & Troubleshooting
The application includes a Debug page to help with technical issues:
- Check the status of saved models
- Create test models if needed
- Test ensemble model creation
- View environment information

---

© 2025 DiabeteXpert. Developed to empower healthcare professionals with AI-driven insights for diabetes risk assessment.
