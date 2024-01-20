import pandas as pd
import numpy as np
from joblib import load
from sklearn import set_config
from interpretation.utils import generate_shap_explanation 
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from os.path import dirname, join, abspath, normpath

# Set transformers output to Pandas DataFrame instead of NumPy array
set_config(transform_output="pandas")


# Current directory
current_dir = dirname(abspath(__file__))
credentials_path = normpath(join(current_dir, "..", "..", "credentials.env"))

_ = load_dotenv(credentials_path)
openai.api_key = os.environ['OPENAI_API_KEY']


# ------- Variables ------- #
global model
global preprocessor
global system_prompt
global query_template


system_prompt = """
You are the assistant of a loan eligibility officer who doesn't know much about machine learning. 
A data scientist built a machine learning model to predict whether or not a loan applicant is eligible for a loan.
You are tasked to explain the model's predictions based on the SHAP (SHapley Additive exPlanations) values of the model's features. 
Your report should focus more on the features that most impacted the model's decision and how they impacted it.
Remember, you are explaining the model's decision to a non-technical person.
"""

query_template = """
Below are the definitions of the features:
- Dependents: Number of dependents of the applicant 
- ApplicantIncome: Income of the applicant
- CoapplicantIncome: Income of the co-applicant
- LoanAmount: Loan amount in thousands
- Loan_Amount_Term: Term of the loan in months
- Gender_Male: 1 if the applicant is a male, 0 otherwise
- Married_Yes: 1 if the applicant is married, 0 otherwise
- Education_Not Graduate: 1 if the applicant is not a graduate, 0 otherwise
- Self_Employed_Yes: 1 if the applicant is self-employed, 0 otherwise
- Property_Area_Rural: 1 if the property is in a rural area, 0 otherwise
- Property_Area_Semiurban: 1 if the property is in a semiurban area, 0 otherwise
- Property_Area_Urban: 1 if the property is in an urban area, 0 otherwise
- Credit_History_1.0: 1 if the applicant has a credit history, 0 otherwise

Below are the names, values, SHAP values, and effects for each prediction in a JSON format:
{explanations_json}

Below is the prediction of the model:
Predicted status: {predicted_status}
Probability of approval: {predicted_proba}%

Based on the information on feature names, values, SHAP values, and effects, generate a report to explain the model's decision.
"""


# Get the paths
current_dir = dirname(abspath(__file__))

model_path = normpath(join(current_dir, "..", "..", "models", "model.pkl"))
preprocessor_path = normpath(join(current_dir, "..", "..", "models", "preprocessor.pkl"))

# ------- Load preprocessor and model------- #
with open(model_path, 'rb') as f:
    model = load(f)

with open(preprocessor_path, 'rb') as f:
    preprocessor = load(f)


# ------- Generate SHAP explanation ------- #
def generate_report(X_test, user_input):
    """
    Generate a report to explain the model's decision.

    This functions takes in the use input data, generates
    shap explanations from that data, convert the explanations to 
    JSON so that we can write a clear prompt for GPT-3.5 and finally
    convert the JSON object to a DataFrame to combine it with the response
    from GPT-3.5. 

    Parameters:
    -----------
    X_test: DataFrame
        Test set
    user_input: DataFrame
        User input
    """
    # Get the SHAP explanation (values and data)
    X_test = preprocessor.transform(X_test)
    shap_explanation = generate_shap_explanation(X_test, user_input)

    # Convert the explanations to an array of structured JSON objects 
    explanation_jsons = explanation_to_json(shap_explanation=shap_explanation,)

    # Predict the status of the loan application
    data = preprocessor.transform(user_input)
    predicted_status = model.predict(data)[0]
    predicted_proba = model.predict_proba(data)[0][1] * 100

    # Create the query
    query = query_template.format(
        explanations_json=explanation_jsons,
        predicted_status=predicted_status,
        predicted_proba=predicted_proba
    )

    # Generate the response
    completion = openai.chat.completions.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": query}
		]
	)
    response = completion.choices[0].message.content
    # response = "This is a test response"

    # Convert the JSON object to a DataFrame
    explanation_df = explanation_to_dataframe(explanation_jsons)

    # Generate the report
    # The report consists of the dataframe as a Markdown table
    # and the response from GPT-3
    report = f"""
## Applicant Information

{explanation_df.to_markdown()}

## Model Decision

{response}
"""

    return report, shap_explanation


# ------- Utility functions ------- #

def explanation_to_json(shap_explanation):
    """
    Create a JSON object from the SHAP explanation
    so that we can easily write a prompt for GPT-3.5.

    Parameters:
    -----------
    feature_names: list
        List of feature names
    shap_explanation: shap._explanation.Explanation
        SHAP explanation object containing the SHAP values and the data

    Returns:
    --------
    explanation_jsons: dict
        A JSON object containing the name of the features,
        their value, their SHAP value and their effect on approval.
    """
    explanation_jsons = []
    feature_names = [
        'Dependents', 'Applicant Income', 'Coapplicant Income', 'Loan Amount',
       'Loan Amount Term', 'Gender', 'Married',
       'Education', 'Self Employed', 'Property Area: Rural',
       'Property Area: Semiurban', 'Property Area: Urban', 'Has Credit History'
    ]
    
    for name, value, shap_value in zip(feature_names, shap_explanation.data.iloc[0].values, shap_explanation.values[0]):        
        explanation_json = {}

        # Map the values to strings for interpretability
        if name == "Gender":
            value = "Male" if value == 1 else "Female"
        elif name == "Married":
            value = "Yes" if value == 1 else "No"
        elif name == "Education":
            value = "Not Graduate" if value == 1 else "Graduate"
        elif name == "Self Employed":
            value = "Yes" if value == 1 else "No"
        elif name == "Property Area: Rural":
            value = "Yes" if value == 1 else "No"
        elif name == "Property Area: Semiurban":
            value = "Yes" if value == 1 else "No"
        elif name == "Property Area: Urban":
            value = "Yes" if value == 1 else "No"
        elif name == "Has Credit History":
            value = "Yes" if value == 1 else "No"

        explanation_json["Name"] = name
        
        # Round numerical features value
        if name in ['Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Loan Amount Term']:
            explanation_json["Value"] = round(value)
        else:
            explanation_json["Value"] = value

        explanation_json["SHAP Value"] = shap_value
        explanation_json["Effect on Approval"] = "Positive" if shap_value > 0 else "Negative"
        explanation_jsons.append(explanation_json)

    return explanation_jsons


def explanation_to_dataframe(explanation_jsons):
    """
    Takes the shap explanation as a JSON object and convert it to a DataFrame.

    Parameters:
    -----------
    explanation_json: dict
        A JSON object representing the name of the feature,
        its value, its SHAP value and its effect on approval.
        Generated by the explanation_to_json function.
    """
    # Convert the JSON object to a DataFrame
    explanation_df = pd.DataFrame(explanation_jsons)

    # Sort feature by impact score
    explanation_df["SHAP Value (Abs)"] = explanation_df["SHAP Value"].abs()
    explanation_df.sort_values(by="SHAP Value (Abs)", ascending=False, inplace=True)
    explanation_df.drop("SHAP Value (Abs)", axis=1, inplace=True)

    # Rename SHAP value to Impact Score to make it more intuitive
    explanation_df.rename(columns={"SHAP Value": "Impact Score"}, inplace=True)

    return explanation_df

