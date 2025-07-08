# random_forest_modeling_tuned.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
import matplotlib.pyplot as plt

# Paths
input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
model_output_path = "/Users/arrryyy/Desktop/bank/outputs/random_forest_tuned.pkl"
conf_matrix_path = "/Users/arrryyy/Desktop/bank/outputs/conf_matrix_rf_tuned.png"

# Load data
df = pd.read_csv(input_path)
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

# Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Random Forest Model:", best_rf)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Save model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
with open(model_output_path, "wb") as f:
    pickle.dump(best_rf, f)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.savefig(conf_matrix_path)
plt.close()

print(f"Model saved at: {model_output_path}")
print(f"Confusion matrix saved at: {conf_matrix_path}")