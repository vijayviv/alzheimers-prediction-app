# 🧠 Alzheimer's Disease Prediction App

This is a machine learning web application that predicts the likelihood of Alzheimer's Disease based on brain MRI features. Built using **Python**, **Streamlit**, and **scikit-learn**, this app provides an easy interface to test new patient data and receive quick predictive results.

---

## 📌 Features

- Predicts Alzheimer's vs. Normal condition using MRI-derived features
- Uses PCA and Ensemble Learning (Random Forest, SVM, Logistic Regression)
- Interactive UI built with Streamlit
- Real-time predictions from uploaded patient data

---

## 📊 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas / NumPy
- PCA for dimensionality reduction
- Ensemble models: Random Forest, SVM, Logistic Regression

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/alzheimers-prediction-app.git
cd alzheimers-prediction-app

alzheimers_app/
├── alzheimers_app.py       # Main Streamlit App
├── model.pkl               # Trained ML model
├── pca.pkl                 # PCA transformer
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

📄 Sample Input Data Format
| Age | Brain Volume | Hippocampus Volume | Entorhinal Cortex Volume | Ventricles Volume | Diagnosis |
| --- | ------------ | ------------------ | ------------------------ | ----------------- | --------- |
| 75  | 1.30e+05     | 4500               | 3300                     | 32000             | AD        |
| 68  | 1.45e+05     | 4900               | 3500                     | 29500             | Normal    |
| 82  | 1.20e+05     | 4200               | 3100                     | 34000             | AD        |
| 70  | 1.38e+05     | 4700               | 3400                     | 31000             | Normal    |



📦 Dataset Information
This model was initially trained on a small sample of 100 data points for testing and prototyping purposes.

The full dataset is included in the /data/ folder (full_dataset.csv) for further training, validation, or benchmarking.

You are encouraged to retrain or fine-tune the model using the complete dataset for better performance and generalization