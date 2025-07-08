import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

# === Paths ===
input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
model_output_path = "/Users/arrryyy/Desktop/bank/outputs/svm_model_tuned.pkl"
conf_matrix_tuned_path = "/Users/arrryyy/Desktop/bank/outputs/conf_matrix_svm_tuned.png"
conf_matrix_basic_path = "/Users/arrryyy/Desktop/bank/outputs/conf_matrix_svm_basic.png"

# === Load Data ===
df = pd.read_csv(input_path)
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 1. Basic SVM (No hyperparameters) ===
print("\n--- Basic Linear SVM ---")
basic_svm = LinearSVC(max_iter=10000, random_state=42)
basic_svm.fit(X_train, y_train)
y_pred_basic = basic_svm.predict(X_test)
acc_basic = accuracy_score(y_test, y_pred_basic)
print(f"Accuracy: {acc_basic:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_basic))

# Confusion Matrix for Basic
cm_basic = confusion_matrix(y_test, y_pred_basic)
disp_basic = ConfusionMatrixDisplay(confusion_matrix=cm_basic, display_labels=basic_svm.classes_)
disp_basic.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix - Basic Linear SVM")
plt.savefig(conf_matrix_basic_path)
plt.close()

# === 2. Hyperparameter Tuning ===
print("\n--- Grid Search with LinearSVC ---")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000]
}
grid_search = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_
y_pred_tuned = best_svm.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {acc_tuned:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_tuned))

# Save best model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
with open(model_output_path, "wb") as f:
    pickle.dump(best_svm, f)

# Confusion Matrix for Tuned SVM
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
disp_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=best_svm.classes_)
disp_tuned.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Tuned Linear SVM")
plt.savefig(conf_matrix_tuned_path)
plt.close()

# === 3. Cross-validation Scores on Tuned Model ===
cv_scores = cross_val_score(best_svm, X, y, cv=5)
print("\n--- Cross-Validation on Tuned SVM ---")
print(f"CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# === Summary ===
print("\n--- Accuracy Comparison ---")
print(f"Basic Linear SVM Accuracy: {acc_basic:.4f}")
print(f"Tuned Linear SVM Accuracy: {acc_tuned:.4f}")
print(f"CV Mean Accuracy (Tuned): {cv_scores.mean():.4f}")