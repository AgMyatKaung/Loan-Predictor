import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess data
data = pd.read_csv("loan1.csv")
X = data[['Annual Salary', 'Bank Balance']]
y = data['Employed']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Perceptron": Perceptron(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC()
}

# Train and compare models
best_model = None
best_accuracy = 0.0
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append((name, accuracy))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save best model
joblib.dump(best_model, "loan_trainModel.pkl")

# Create Streamlit app
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

    st.write("## Model Comparison Results")
    df_results = pd.DataFrame(results, columns=["Model", "Accuracy"])
    st.table(df_results)

if __name__ == "__main__":
    main()
