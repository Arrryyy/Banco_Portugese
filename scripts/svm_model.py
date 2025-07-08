# svm_modeling_tuned.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
import matplotlib.pyplot as plt

# Paths
input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
model_output_path = "/Users/arrryyy/Desktop/bank/outputs/svm_model_tuned.pkl"
conf_matrix_path = "/Users/arrryyy/Desktop/bank/outputs/conf_matrix_svm.png"

# Load data
df = pd.read_csv(input_path)
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Perform grid search
grid_search = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best SVM Model:", best_svm)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Save model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
with open(model_output_path, "wb") as f:
    pickle.dump(best_svm, f)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svm.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM")
plt.savefig(conf_matrix_path)
plt.close()

print(f"Model saved at: {model_output_path}")
print(f"Confusion matrix saved at: {conf_matrix_path}")