import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

# Paths
input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
model_output_path = "/Users/arrryyy/Desktop/bank/outputs/random_forest_model.pkl"
conf_matrix_path = "/Users/arrryyy/Desktop/bank/outputs/confusion_matrix_rf.png"

# Load the dataset
df = pd.read_csv(input_path)

# Features and target
X = df.drop("y", axis=1)
y = df["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest with class_weight balanced to handle imbalance
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Random Forest Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# Save the model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
with open(model_output_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Model saved to {model_output_path}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest Model")
plt.savefig(conf_matrix_path)
plt.close()
print(f"Confusion matrix saved to {conf_matrix_path}")