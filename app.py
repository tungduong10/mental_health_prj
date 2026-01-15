import streamlit as st
import pandas as pd
import joblib

#load the trained pipeline (preprocessor + model)
with open('models/model_pipeline.pkl', 'rb') as f:
    model_pipeline = joblib.load(f)

# Load expected feature names for validation
try:
    with open('models/feature_names.pkl', 'rb') as f:
        expected_features = joblib.load(f)
except:
    expected_features = None

st.title("🧠 Mental Health Treatment Predictor")

st.markdown("""
This tool predicts the likelihood that a person working in tech would seek mental health treatment, based on workplace and personal factors.
""")

#collect user inputs
age = st.slider("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
self_employed = st.selectbox("Are you self-employed?", ["Yes", "No", "Unknown or other"])
family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
work_interfere = st.selectbox("Does your mental health affect your work?", ["Never", "Rarely", "Sometimes", "Often", "Unknown or other"])
no_employees = st.selectbox("Company size?(personnel)", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
benefits = st.selectbox("Mental health benefits provided?", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Access to care options at work?", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Wellness program available?", ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Encouraged to seek help?", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is your anonymity secured?", ["Yes", "No", "Don't know"])
leave = st.selectbox("Ease of taking medical leave?", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
mental_health_consequence = st.selectbox("Perceived mental health consequences at work?", ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Perceived physical health consequences at work?", ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Are your coworkers supportive?", ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Are your supervisors supportive?", ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Was mental health discussed in job interview?", ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Was physical health discussed in interview?", ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Company view of mental health vs physical health?", ["Yes", "No", "Don't know"])

#prepare input dict
input_data = {
    'Age': age,
    'Gender': gender,
    'self_employed': self_employed,
    'family_history': family_history,
    'work_interfere': work_interfere,
    'no_employees': no_employees,
    'remote_work': remote_work,
    'benefits': benefits,
    'care_options': care_options,
    'wellness_program': wellness_program,
    'seek_help': seek_help,
    'anonymity': anonymity,
    'leave': leave,
    'mental_health_consequence': mental_health_consequence,
    'phys_health_consequence': phys_health_consequence,
    'coworkers': coworkers,
    'supervisor': supervisor,
    'mental_health_interview': mental_health_interview,
    'phys_health_interview': phys_health_interview,
    'mental_vs_physical': mental_vs_physical
}

#convert to DataFrame for model compatibility
input_df = pd.DataFrame([input_data])

# Validate features match training data
if expected_features and set(input_df.columns) != set(expected_features):
    st.error("⚠️ Input features don't match training data. Please check your inputs.")
else:
    #predict
    if st.button("Predict"):
        prediction = model_pipeline.predict(input_df)[0]  # returns 1 for Yes, 0 for No
        prediction_label = "Yes" if prediction == 1 else "No"
        if prediction == 1:
            st.subheader("✅ Likely to Seek Treatment")
        else:
            st.subheader("❌ Not Likely to Seek Treatment")