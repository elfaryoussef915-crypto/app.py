# Multi-Disease Diagnostic System 🏥

An AI-powered web application built with Streamlit that predicts the likelihood of Diabetes, Heart Disease, and Parkinson's Disease based on clinical metrics and vocal features.

## Features
- **Three Diagnostic Models**:
  - **Diabetes**: Predicts risk based on pregnancies, glucose, BMI, etc.
  - **Heart Disease**: Predicts risk based on cardiovascular metrics like cholesterol and heart rate.
  - **Parkinson's Disease**: Analyzes 22 vocal features to detect potential symptoms.
- **Explainable AI (XAI)**: Uses **SHAP (SHapley Additive exPlanations)** to show exactly which features influenced each prediction.
- **Responsive UI**: Clean, modern interface with interactive inputs and tooltips.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Files
- `app.py`: Main Streamlit application.
- `diabetes_model.sav`: Trained SVC model for diabetes prediction.
- `heart_disease_model.sav`: Trained Logistic Regression model for heart disease.
- `parkinsons_model.sav`: Trained SVC model for Parkinson's prediction.
- `requirements.txt`: Python dependencies.

## Disclaimer
This application is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis or advice.
