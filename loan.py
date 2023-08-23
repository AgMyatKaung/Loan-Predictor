import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

model = joblib.load("loan_trainModel.pkl")

def predict_bank_balance(salary):
    predicted_balance = model.predict([[salary]])
    return predicted_balance[0]

def main():
    st.title("Bank Balance Prediction")

    st.write("Enter the employee's annual salary:")
    salary = st.number_input("Annual Salary", min_value=0)

    if st.button("Predict Bank Balance"):
        predicted_balance = predict_bank_balance(salary)
        st.success(f"Predicted Bank Balance: {predicted_balance:.2f}")

if __name__ == "__main__":
    main()


