# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import scipy.special as sc
import warnings

warnings.filterwarnings("ignore")

# ---------- Configuration & Styling ----------
st.set_page_config(
    page_title="Multi-Disease Diagnostic System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- Helper Functions ----------
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

def get_prediction_data(model, X):
    pred = model.predict(X)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X)[0]
        prob = sc.expit(d)
    return pred, prob

def explain_prediction(model, X, feature_names):
    try:
        # Use a small sample for background if possible, otherwise use X itself
        # For simple models like LogisticRegression or SVC, KernelExplainer or LinearExplainer might be better
        # but SHAP's Explainer often handles them automatically.
        # Note: SHAP can be slow for SVC.
        explainer = shap.Explainer(model.predict, X, silent=True)
        shap_values = explainer(X)
        return shap_values
    except Exception as e:
        return None

def display_explanation(model, X, feature_names):
    st.subheader("🔍 Prediction Insights (Explainable AI)")

    with st.spinner("Generating explanation..."):
        shap_values = explain_prediction(model, X, feature_names)

    if shap_values is not None and len(shap_values.values) > 0:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**Feature Impact (SHAP Waterfall)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        with col2:
            st.write("**Top Contributing Factors**")
            vals = shap_values.values[0]
            abs_vals = np.abs(vals)
            pct = 100 * abs_vals / np.sum(abs_vals) if np.sum(abs_vals) != 0 else np.zeros_like(abs_vals)
            df_impact = pd.DataFrame({
                "Feature": feature_names,
                "Impact Value": vals,
                "Importance %": pct
            }).sort_values("Importance %", ascending=False)

            st.dataframe(df_impact.style.format({"Impact Value": "{:.4f}", "Importance %": "{:.1f}%"}))

            main_feat = df_impact.iloc[0]
            direction = "increases" if main_feat["Impact Value"] > 0 else "decreases"
            st.info(f"**Insight:** '{main_feat['Feature']}' is the most significant factor here, which {direction} the risk.")
    else:
        st.warning("Could not generate SHAP explanation. Displaying model coefficients instead.")
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            impact = X.values[0] * coefs
            df_coef = pd.DataFrame({"Feature": feature_names, "Coefficient Impact": impact})
            st.table(df_coef.sort_values("Coefficient Impact", key=abs, ascending=False))

# ---------- Load Models ----------
diabetes_model = load_model("diabetes_model.sav")
heart_model = load_model("heart_disease_model.sav")
parkinsons_model = load_model("parkinsons_model.sav")

# ---------- Sidebar Navigation ----------
st.sidebar.title("🏥 Health Diagnostics")
choice = st.sidebar.radio("Navigate", ["Home", "Diabetes Check", "Heart Disease Check", "Parkinson's Check"])

# ---------- UI Pages ----------

if choice == "Home":
    st.title("AI-Powered Disease Prediction System")
    st.markdown("""
    ### Welcome to the Smart Health Assistant
    This platform utilizes machine learning models to provide preliminary health assessments for three major conditions.

    #### Key Features:
    - **Diabetes Prediction**: Based on clinical metrics like Glucose and BMI.
    - **Heart Disease Prediction**: Uses cardiovascular data to assess risk.
    - **Parkinson's Prediction**: Analyzes vocal features to detect early signs.
    - **Explainable AI**: Every prediction comes with a 'Why', showing which factors influenced the model's decision using SHAP.

    ---
    *Disclaimer: This tool is for educational purposes and should not replace professional medical advice.*
    """)

    cols = st.columns(3)
    with cols[0]:
        st.info("📊 **Reliable Models**\nTrained on standard medical datasets.")
    with cols[1]:
        st.success("🧠 **Explainable Results**\nUnderstand the 'Why' behind every result.")
    with cols[2]:
        st.warning("⚡ **Instant Feedback**\nGet results in seconds.")

elif choice == "Diabetes Check":
    st.title("Diabetes Risk Assessment")

    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            preg = st.number_input("Pregnancies", 0, 20, 0, help="Number of times pregnant")
            glucose = st.number_input("Glucose Level", 0, 500, 120, help="Plasma glucose concentration (mg/dL)")
            bp = st.number_input("Blood Pressure", 0, 200, 70, help="Diastolic blood pressure (mm Hg)")
            skin = st.number_input("Skin Thickness", 0, 100, 20, help="Triceps skin fold thickness (mm)")
        with col2:
            insulin = st.number_input("Insulin Level", 0, 900, 80, help="2-Hour serum insulin (mu U/ml)")
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0, help="Body mass index (weight in kg/(height in m)^2)")
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, help="Diabetes pedigree function (genetic risk)")
            age = st.number_input("Age", 1, 120, 30)

        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        X = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]], columns=feature_names)

        if st.button("Analyze Risk"):
            if diabetes_model:
                pred, prob = get_prediction_data(diabetes_model, X)
                if pred == 1:
                    st.error(f"### Result: High Risk of Diabetes (Probability: {prob*100:.1f}%)")
                else:
                    st.success(f"### Result: Low Risk of Diabetes (Probability: {prob*100:.1f}%)")

                display_explanation(diabetes_model, X, feature_names)
            else:
                st.error("Model file missing.")
        st.markdown('</div>', unsafe_allow_html=True)

elif choice == "Heart Disease Check":
    st.title("Cardiovascular Health Check")

    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
            trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
        with c2:
            chol = st.number_input("Serum Cholestoral (mg/dl)", 50, 600, 230)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x==1 else "False")
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
            thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
        with c3:
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
            ca = st.selectbox("Major Vessels Colored by Flourosopy", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], help="1 = normal; 2 = fixed defect; 3 = reversable defect")

        feat_names_heart = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        Xh = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=feat_names_heart)

        if st.button("Analyze Cardiac Health"):
            if heart_model:
                pred, prob = get_prediction_data(heart_model, Xh)
                if pred == 1:
                    st.error(f"### Result: Heart Disease Detected (Probability: {prob*100:.1f}%)")
                else:
                    st.success(f"### Result: No Significant Heart Disease Detected (Probability: {prob*100:.1f}%)")

                display_explanation(heart_model, Xh, feat_names_heart)
            else:
                st.error("Model file missing.")
        st.markdown('</div>', unsafe_allow_html=True)

elif choice == "Parkinson's Check":
    st.title("Parkinson's Vocal Analysis")
    st.info("This model analyzes 22 acoustic features from voice recordings to predict Parkinson's disease.")

    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        tabs = st.tabs(["Frequency Features", "Jitter & Shimmer", "Acoustic Metrics", "Non-linear Measures"])

        with tabs[0]:
            col1, col2 = st.columns(2)
            fo = col1.number_input("MDVP:Fo(Hz)", value=119.9, help="Average vocal fundamental frequency")
            fhi = col2.number_input("MDVP:Fhi(Hz)", value=157.3, help="Maximum vocal fundamental frequency")
            flo = col1.number_input("MDVP:Flo(Hz)", value=75.0, help="Minimum vocal fundamental frequency")

        with tabs[1]:
            c1, c2 = st.columns(2)
            jitter = c1.number_input("MDVP:Jitter(%)", value=0.0078, format="%.5f")
            jitter_abs = c2.number_input("MDVP:Jitter(Abs)", value=0.00007, format="%.5f")
            rap = c1.number_input("MDVP:RAP", value=0.0037, format="%.5f")
            ppq = c2.number_input("MDVP:PPQ", value=0.0055, format="%.5f")
            ddp = c1.number_input("Jitter:DDP", value=0.011, format="%.5f")
            shimmer = c2.number_input("MDVP:Shimmer", value=0.044, format="%.5f")
            shimmer_db = c1.number_input("MDVP:Shimmer(dB)", value=0.426, format="%.5f")
            apq3 = c2.number_input("Shimmer:APQ3", value=0.02, format="%.5f")
            apq5 = c1.number_input("Shimmer:APQ5", value=0.03, format="%.5f")
            apq = c2.number_input("MDVP:APQ", value=0.03, format="%.5f")
            dda = c1.number_input("Shimmer:DDA", value=0.06, format="%.5f")

        with tabs[2]:
            nhr = st.number_input("NHR (Noise-to-Harmonics)", value=0.022)
            hnr = st.number_input("HNR (Harmonics-to-Noise)", value=21.0)

        with tabs[3]:
            rpde = st.number_input("RPDE", value=0.41)
            dfa = st.number_input("DFA", value=0.81)
            spread1 = st.number_input("spread1", value=-4.8)
            spread2 = st.number_input("spread2", value=0.26)
            d2 = st.number_input("D2", value=2.3)
            ppe = st.number_input("PPE", value=0.28)

        feat_names_p = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
                      'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3',
                      'Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']

        Xp = pd.DataFrame([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]], columns=feat_names_p)

        if st.button("Analyze Vocal Patterns"):
            if parkinsons_model:
                pred, prob = get_prediction_data(parkinsons_model, Xp)
                if pred == 1:
                    st.error(f"### Result: Parkinson's Patterns Detected (Probability: {prob*100:.1f}%)")
                else:
                    st.success(f"### Result: Healthy Vocal Patterns (Probability: {prob*100:.1f}%)")

                display_explanation(parkinsons_model, Xp, feat_names_p)
            else:
                st.error("Model file missing.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.write("---")
st.caption("Built with ❤️ using Streamlit, Scikit-Learn, and SHAP.")
