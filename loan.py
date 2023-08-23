import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("D:\Programming\AI\loan\Loan1.csv")
X = data[['Annual Salary', 'Bank Balance']]
y = data['Employed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Perceptron": Perceptron(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC()
}

best_model = None
best_accuracy = 0.0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

joblib.dump(best_model, "D:\Programming\AI\loan\loan_trainModel.pkl")

def predict_loan_eligibility(salary, balance):
    features = scaler.transform([[salary, balance]])
    prediction = best_model.predict(features)
    return prediction[0]

def main():
    st.title("Loan Eligibility Prediction")

    st.write("Enter the employee's annual salary and bank balance:")
    salary = st.number_input("Annual Salary", min_value=0)
    balance = st.number_input("Bank Balance", min_value=0)

    if st.button("Predict Loan Eligibility"):
        eligibility = predict_loan_eligibility(salary, balance)
        if eligibility == 1:
            st.success("Employee is eligible for a loan.")
        else:
            st.warning("Employee is not eligible for a loan.")

if __name__ == "__main__":
    main()


