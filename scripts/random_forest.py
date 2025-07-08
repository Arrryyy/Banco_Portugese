import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

# === Paths ===
input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
model_output_path = "/Users/arrryyy/Desktop/bank/outputs/random_forest_tuned.pkl"
conf_matrix_basic_path = "/Users/arrryyy/Desktop/bank/outputs/conf_matrix_rf_basic.png"
conf_matrix_tuned_path = "/Users/arrryyy/Desktop/bank/outputs/conf_matrix_rf_tuned.png"

# === Load Data ===
df = pd.read_csv(input_path)
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 1. Basic Random Forest ===
print("\n--- Basic Random Forest ---")
basic_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
basic_rf.fit(X_train, y_train)
y_pred_basic = basic_rf.predict(X_test)
acc_basic = accuracy_score(y_test, y_pred_basic)
print(f"Accuracy: {acc_basic:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_basic))

# Confusion Matrix (Basic)
cm_basic = confusion_matrix(y_test, y_pred_basic)
disp_basic = ConfusionMatrixDisplay(confusion_matrix=cm_basic, display_labels=basic_rf.classes_)
disp_basic.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix - Basic Random Forest")
plt.savefig(conf_matrix_basic_path)
plt.close()

# === 2. Hyperparameter Tuning ===
print("\n--- Grid Search: Tuned Random Forest ---")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {acc_tuned:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_tuned))

# Save model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
with open(model_output_path, "wb") as f:
    pickle.dump(best_rf, f)

# Confusion Matrix (Tuned)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
disp_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=best_rf.classes_)
disp_tuned.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Tuned Random Forest")
plt.savefig(conf_matrix_tuned_path)
plt.close()

# === 3. Cross-validation on Tuned Model ===
cv_scores = cross_val_score(best_rf, X, y, cv=5)
print("\n--- Cross-Validation on Tuned Random Forest ---")
print(f"CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# === Summary ===
print("\n--- Accuracy Comparison ---")
print(f"Basic RF Accuracy: {acc_basic:.4f}")
print(f"Tuned RF Accuracy: {acc_tuned:.4f}")
print(f"Tuned RF CV Accuracy: {cv_scores.mean():.4f}")

