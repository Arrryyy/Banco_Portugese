from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
df = pd.read_csv(input_path)
X = df.drop("y", axis=1)
y = df["y"]

# Reduce data size temporarily
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))