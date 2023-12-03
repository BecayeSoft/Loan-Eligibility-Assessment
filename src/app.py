import pandas as pd
import matplotlib.pyplot as plt
import shap
from joblib import load
import streamlit as st
from sklearn import set_config
from reporting.utils import generate_report
from os.path import dirname, join, abspath, normpath

# Set transformers output to Pandas DataFrame instead of NumPy array
set_config(transform_output="pandas")


# ------- Variables ------- #
global model
global preprocessor


# Get the paths
current_dir = dirname(abspath(__file__))
model_path = normpath(join(current_dir, "..", "models", "model.pkl"))
preprocessor_path = normpath(join(current_dir, "..", "models", "preprocessor.pkl"))


# ------- Load preprocessor and model------- #
with open(model_path, 'rb') as f:
    model = load(f)

with open(preprocessor_path, 'rb') as f:
    preprocessor = load(f)


def load_test_data():
    """Load test data"""
    # Current directory
    current_dir = dirname(abspath(__file__))
    test_path = normpath(join(current_dir, "..", "data", "raw", "loan-data-test.csv"))
    X_test = pd.read_csv(test_path)

    # Drop the Loan_ID column
    X_test.drop(columns=['Loan_ID'], inplace=True)
    
    # Mapping number of dependents to numerical values
    X_test.Dependents.replace('3+', 3, inplace=True)

    return X_test


def predict_loan_eligibility(data):
    """Predict loan eligibility based on user input"""
    data = preprocessor.transform(data)
    prediction = model.predict(data)
    return prediction


def main():
    """Loan Eligibility Prediction App"""
    st.title("Loan Eligibility Assessment")

    # Get user input
    st.sidebar.header("Applicant Data")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Married", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3 or more"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    applicantIncome = st.sidebar.slider("ApplicantIncome", 150, 81000, 250)
    coapplicantIncome = st.sidebar.slider("Coapplicant Income", 0, 41667, 0)
    loanAmount = st.sidebar.slider("Loan Amount", 9, 700, 100)
    loan_Amount_Term = st.sidebar.slider("Loan Amount Term", 12, 480, 360)
    credit_History = st.sidebar.selectbox("Has Credit History", ["Yes", "No"])      # TODO convert to int
    property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Mapping user input to numerical values
    credit_History = 1 if credit_History == "Yes" else 0
    dependents = 3 if dependents == "3 or more" else dependents

    # Create a dictionary to store user input
    user_input = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_Employed,
        'ApplicantIncome': applicantIncome,
        'CoapplicantIncome': coapplicantIncome,
        'LoanAmount': loanAmount,
        'Loan_Amount_Term': loan_Amount_Term,
        'Credit_History': credit_History,
        'Property_Area': property_Area
    }


    # Convert user input to DataFrame for prediction
    input_df = pd.DataFrame([user_input])

    # Predict the status of the loan application
    prediction = predict_loan_eligibility(input_df)

    # # Display prediction
    # st.write("### Decision")

    if prediction == 1:
        st.write("Congratulations 🎉")
        st.write("Your loan application has been approved! Find more details below.")
    else:
        st.write("Your loan will need further investigation before it can be approved 🙂")
        st.write("Please find more details below or contact us for more information.")


    # Generate report
    X_test = load_test_data()
    report, shap_explanation = generate_report(X_test, user_input=input_df)
    st.write(report)

    # Display the waterfall plot
    st.write("### Impact of each variable on the prediction")
    st.write("On the graph below, the red bars indicate positive contributions to the approval of \
        the loan application, while the blue ones signify contributions leading to rejection.")
    fig = plt.figure()
    shap.plots.waterfall(shap_explanation[0], show=False)
    st.pyplot(fig)


if __name__ == '__main__':
    main()

