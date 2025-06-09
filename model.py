import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
columns = [
    "Season", "Age", "Childish diseases", "Accident trauma", "Surgical intervention",
    "High fever last year", "Frequency of alcohol consumption",
    "Smoking habit", "Number of hours spent sitting per day", "Output"
]
df = pd.read_csv("C:/mlcas/fertility_Diagnosis.txt", header=None, names=columns)

# Encode target
df['Output'] = df['Output'].map({'N': 0, 'O': 1})  # N: Normal, O: Altered

# Split data
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fertility_model.pkl")
print("✅ Model saved to fertility_model.pkl")
