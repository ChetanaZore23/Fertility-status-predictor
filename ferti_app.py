import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("🌱 FertiCheck: Fertility Status Predictor")

st.sidebar.header("Input Your Information")
season = st.sidebar.slider("Season (Spring=0 to Winter=3)", 0, 3, 0)
age = st.sidebar.slider("Age", 18, 36, 25)
child_disease = st.sidebar.selectbox("Had Childhood Diseases?", [0, 1])
trauma = st.sidebar.selectbox("Had Trauma?", [0, 1])
surgery = st.sidebar.selectbox("Had Surgical Intervention?", [0, 1])
fever = st.sidebar.selectbox("High Fever in Last Year?", [0, 1])
alcohol = st.sidebar.slider("Alcohol Consumption (0-4)", 0, 4, 1)
smoking = st.sidebar.selectbox("Smoking Habit?", [0, 1])
sitting_hours = st.sidebar.slider("Hours Sitting per Day", 0, 16, 4)

# Prepare input
input_data = [[season, age, child_disease, trauma, surgery, fever, alcohol, smoking, sitting_hours]]
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][prediction]

# Output result
st.subheader("🔍 Prediction Result")
result = "Normal Fertility" if prediction == 0 else "Altered Fertility"
st.write(f"### 🧪 You are likely to have: **{result}** ({prob*100:.1f}% confidence)")
