# svm_modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
model_output_path = "/Users/arrryyy/Desktop/bank/outputs/svm_model.pkl"

df = pd.read_csv(input_path)

# Feature matrix (X) and target vector (y)
X = df.drop('y', axis=1)
y = df['y']
print(X.shape)
print(y.shape)

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = LinearSVC(C=1.0, random_state=42, max_iter=10000)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("SVM Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)

os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
with open(model_output_path, 'wb') as f:
    pickle.dump(svm_model, f)

print(f"Model saved as pickle to {model_output_path}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)

output_dir = os.path.dirname(model_output_path)
cm_output_path = os.path.join(output_dir, "confusion_matrix.png")
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM Model")
plt.savefig(cm_output_path)
plt.close()

print(f"Confusion matrix saved to {cm_output_path}")