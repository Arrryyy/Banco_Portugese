import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Paths
input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
model_output_path = "/Users/arrryyy/Desktop/bank/outputs/logreg_model.pkl"
conf_matrix_output_path = "/Users/arrryyy/Desktop/bank/outputs/logreg_confusion_matrix.png"

# Load dataset
df = pd.read_csv(input_path)

# Features and target
X = df.drop('y', axis=1)
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

y_pred = logreg_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)
print("\nClassification Report:\n", report)


os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
with open(model_output_path, 'wb') as f:
    pickle.dump(logreg_model, f)
print(f"Model saved as pickle to {model_output_path}")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg_model.classes_)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig(conf_matrix_output_path)
plt.close()
print(f"Confusion matrix saved to {conf_matrix_output_path}")