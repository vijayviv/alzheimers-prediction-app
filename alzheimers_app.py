import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


st.set_page_config(page_title="Alzheimer's Predictor", layout="wide")
st.title("👤 VIJAY V(31182310405)")
st.title("🧠 Alzheimer's Disease Prediction")
st.write("Enter patient details below to predict Alzheimer's Disease status.")


df = pd.read_csv("alzheimers_vijay.csv")
df = df.drop(columns=["PatientID", "DoctorInCharge"])
label_column = "Diagnosis"


feature_columns = [col for col in df.columns if col != label_column]
X = df[feature_columns]
y = df[label_column].astype("category")
y_encoded = y.cat.codes
label_map = dict(enumerate(y.cat.categories))


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
logreg = LogisticRegression(max_iter=1000, random_state=42)

model = VotingClassifier(estimators=[
    ("rf", rf),
    ("svm", svm),
    ("logreg", logreg)
], voting="soft")

model.fit(X_pca, y_encoded)


st.subheader("🔢 Patient Information")

input_values = {}
cols = st.columns(4)
for i, col in enumerate(feature_columns):
    with cols[i % 4]:
        default_val = float(df[col].mean())
        input_val = st.number_input(col, value=default_val)
        input_values[col] = input_val
label_map = {
    0: "Alzheimer",
    1: "Normal"
}

if st.button("Predict"):
    input_df = pd.DataFrame([input_values])
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)[0]
    predicted_label = label_map[prediction]

    if "alzheimer" in predicted_label.lower() or "ad" in predicted_label.lower():
        st.success("🧠 Predicted: **AD (Alzheimer's Disease)**")
    else:
        st.success("✅ Predicted: **Normal**")