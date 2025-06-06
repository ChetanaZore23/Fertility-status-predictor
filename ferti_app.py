import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Title
st.title("🌱 FertiCheck: Fertility Status Predictor")

# Load dataset
@st.cache_data
def load_data():
    columns = [
        "Season", "Age", "Childish diseases", "Accident trauma", "Surgical intervention",
        "High fever last year", "Frequency of alcohol consumption",
        "Smoking habit", "Number of hours spent sitting per day", "Output"
    ]
    df = pd.read_csv("fertility_Diagnosis.txt", header=None, names=columns)
    return df

df = load_data()

# Encode target
df['Output'] = df['Output'].map({'N': 0, 'O': 1})  # N: Normal, O: Altered

st.subheader("📋 Dataset Preview")
st.write(df.head())

# Split data
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sidebar input
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

# Predict
input_data = [[season, age, child_disease, trauma, surgery, fever, alcohol, smoking, sitting_hours]]
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][prediction]

# Show result
st.subheader("🔍 Prediction Result")
result = "Normal Fertility" if prediction == 0 else "Altered Fertility"
st.write(f"### 🧪 You are likely to have: **{result}** ({prob*100:.1f}% confidence)")

# Metrics (optional)
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=["Normal", "Altered"]))
