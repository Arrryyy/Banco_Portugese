import sys
import pickle
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QGridLayout, QMessageBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

class SVMPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📈 Bank SVM Predictor")
        self.setFixedSize(500, 500)
        self.initUI()

        # Load the trained model
        with open("/Users/arrryyy/Desktop/bank/outputs/svm_model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def initUI(self):
        layout = QVBoxLayout()

        title = QLabel("🔍 Bank SVM Predictor")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(10)

        # Inputs with better labels
        self.age_input = QLineEdit()
        self.balance_input = QLineEdit()

        self.job_input = QComboBox()
        self.job_options = ["blue-collar", "entrepreneur", "housemaid", "management",
                            "retired", "self-employed", "services", "student", "technician",
                            "unemployed", "unknown"]
        self.job_input.addItems(self.job_options)

        self.marital_input = QComboBox()
        self.marital_options = ["married", "single", "divorced"]
        self.marital_input.addItems(self.marital_options)

        self.education_input = QComboBox()
        self.education_options = ["secondary", "tertiary", "unknown", "primary"]
        self.education_input.addItems(self.education_options)

        self.default_input = QComboBox()
        self.default_options = ["no", "yes"]
        self.default_input.addItems(self.default_options)

        self.housing_input = QComboBox()
        self.housing_options = ["no", "yes"]
        self.housing_input.addItems(self.housing_options)

        self.loan_input = QComboBox()
        self.loan_options = ["no", "yes"]
        self.loan_input.addItems(self.loan_options)

        inputs = [
            ("Age", self.age_input),
            ("Balance (€)", self.balance_input),
            ("Job", self.job_input),
            ("Marital Status", self.marital_input),
            ("Education", self.education_input),
            ("Default", self.default_input),
            ("Housing Loan", self.housing_input),
            ("Personal Loan", self.loan_input)
        ]

        for i, (label_text, widget) in enumerate(inputs):
            label = QLabel(label_text)
            label.setFont(QFont("Arial", 10))
            grid.addWidget(label, i, 0)
            grid.addWidget(widget, i, 1)

        layout.addLayout(grid)

        # Predict button
        predict_button = QPushButton("🔮 Predict")
        predict_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        predict_button.clicked.connect(self.predict)
        layout.addWidget(predict_button)

        # Result display
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict(self):
        try:
            age = float(self.age_input.text())
            balance = float(self.balance_input.text())

            # Encode inputs
            encoded = [age, balance]
            job_encoded = [1 if self.job_input.currentText() == job else 0 for job in self.job_options]
            marital_encoded = [1 if self.marital_input.currentText() == "married" else 0,
                               1 if self.marital_input.currentText() == "single" else 0]
            education_encoded = [1 if self.education_input.currentText() == "secondary" else 0,
                                 1 if self.education_input.currentText() == "tertiary" else 0,
                                 1 if self.education_input.currentText() == "unknown" else 0]
            default_encoded = [1 if self.default_input.currentText() == "yes" else 0]
            housing_encoded = [1 if self.housing_input.currentText() == "yes" else 0]
            loan_encoded = [1 if self.loan_input.currentText() == "yes" else 0]

            input_array = np.array([encoded + job_encoded + marital_encoded +
                                    education_encoded + default_encoded +
                                    housing_encoded + loan_encoded])

            prediction = self.model.predict(input_array)[0]
            result_text = "✅ Prediction: YES" if prediction == 1 else "❌ Prediction: NO"
            self.result_label.setText(result_text)

        except Exception as e:
            self.result_label.setText("⚠️ Error: Invalid input")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SVMPredictorApp()
    window.show()
    sys.exit(app.exec())