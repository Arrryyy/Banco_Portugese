from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
input_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"
df = pd.read_csv(input_path)
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

print("LDA Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))