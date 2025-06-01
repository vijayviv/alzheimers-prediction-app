import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("alzheimers_vijay.csv")
print("Dataset Shape:", df.shape)
print("Column Names:", df.columns.tolist())

df = df.drop(columns=["PatientID", "DoctorInCharge"])  # Drop irrelevant ID/name columns

label_column = 'Diagnosis'

X = df.drop(label_column, axis=1)
y = df[label_column]

y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
logreg = LogisticRegression(max_iter=1000, random_state=42)

ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('svm', svm),
    ('logreg', logreg)
], voting='soft')

ensemble.fit(X_train_pca, y_train)

y_pred = ensemble.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()