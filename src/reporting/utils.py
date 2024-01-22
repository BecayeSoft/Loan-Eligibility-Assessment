import pandas as pd
from joblib import load
from sklearn import set_config
from interpretation.utils import generate_shap_explanation 
import openai
import streamlit as st
from os.path import dirname, join, abspath, normpath

# Set transformers output to Pandas DataFrame instead of NumPy array
set_config(transform_output="pandas")

# Load the API key from streamlit secrets
openai.api_key = st.secrets.OPENAI_API_KEY


# ------- Variables ------- #
global model
global preprocessor
global system_prompt
global query_template



system_prompt = """
The system evaluates loan applications using applicant data. 
You need to explain the system's decision, considering features and their impacts, and this explanation is tailored for the non-technical applicant. 
No greetings or closings are necessary. 
Emphasize the features that had the most influence on the system's decision and how they affected that decision.
When you mention a feature, include the feature's name and value.
Use the term "system" to reference the model and avoid technical jargon related to the SHAP values.

IMPORTANT
---------
Higher ApplicantIncome, CoapplicantIncome and LoanAmount are associated with a higher probability of approval. 
Higher LoanAmount and Loan_Amount_Term are associated with a lower probability of approval.
Loan Amount ranges from $9 to $700 (in thousands).
Loan Amount Term ranges from 12 to 480 months.
"""

response_template = """
Your loan application has been approved. Several factors contributed to this decision.

### What you did well:
- **Income**: You have an income of \$3,235. This factor significantly boosts your chances of approval as a higher income increases the likelihood of getting the loan approved.
- **Co-applicant's Income**: You have a co-applicant with an income of \$2. This factor significantly boosts your chances of approval, as a higher co-applicant income increases the likelihood of getting the loan approved.
- **Requested Loan Amount:** Your loan request of \$77,000 falls within the lower range of our allowable amount, which spans from \$9,000 to \$700,000. This contributed positively to the approval decision.
- **Credit History:** You have a credit history, which is required for loan approval.
- Etc.

### What you need to work on:
- **Loan Term Duration:** The chosen loan term of 360 months (30 years) exceeds the midpoint in our range of 12 to 480 months. Opting for a longer loan term slightly diminishes your chances of approval.
- Etc.
...
"""

query_template = """
Below are the definitions of the features:
- Dependents: Number of dependents of the applicant 
- ApplicantIncome: Income of the applicant
- CoapplicantIncome: Income of the co-applicant
- LoanAmount: Loan amount 
- Loan_Amount_Term: Term of the loan in months
- Gender: then gender of the applicant
- Self Employed: wheather the applicant is self-employed or not
- Property Area:Rural: "Yes" if the property is in a rural area, "No" otherwise
- PropertyArea: Semiurban: "Yes" if the property is in a semiurban area, "No" otherwise
- Property_Area: Urban: "Yes" if the property is in an urban area, "No" otherwise
- Has Credit History: "Yes" if the applicant has a credit history, "No" otherwise

Below are the names, values, SHAP values, and effects for each prediction in a JSON format:
{explanation_jsons}

Below is the prediction of the model:
Predicted status: {predicted_status}
Probability of approval: {predicted_proba}%

-----
Based on the information on feature names, values, SHAP values, and effects, 
generate a report to explain the model's decision in simple terms.
Below is an example of response so that you can get the pattern.
Rewrite it to fit the current context based on the information above:
The bulleted list should be ordered by impact magnitude.
{response_template}

Conclude with a summary of the most important factors and their effects on the decision.

Recommend actions to improve the chances of approval.
"""


# Get the paths
current_dir = dirname(abspath(__file__))
model_path = normpath(join(current_dir, "..", "streamlit-prod", "model.pkl"))
preprocessor_path = normpath(join(current_dir, "..", "streamlit-prod", "preprocessor.pkl"))

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
    explanation_jsons = explanation_to_json(shap_explanation)

    # Predict the status of the loan application
    data = preprocessor.transform(user_input)
    predicted_proba = model.predict_proba(data)[0][1] * 100
    predicted_status = model.predict(data)[0]
    predicted_status = "approved" if predicted_status == 1 else "rejected"

    # Create the query
    query = query_template.format(
        explanation_jsons=explanation_jsons,
        predicted_status=predicted_status,
        predicted_proba=predicted_proba,
        response_template=response_template
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
    # # response = "This is a test response"
    # response = query

    # Convert the JSON object to a DataFrame
    explanation_df = explanation_to_dataframe(explanation_jsons)

    # Generate the report
    # The report consists of the dataframe as a Markdown table
    # and the response from GPT-3
    report = f"""
### Summary

{explanation_df.to_markdown()}

---
## Explanation

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
        'Loan Term', 'Gender', 
        # 'Married', 'Education', 
        'Self Employed', 'Property Area: Rural',
        'Property Area: Semiurban', 'Property Area: Urban', 'Has Credit History'
    ]
    
    for name, value, shap_value in zip(feature_names, 
        shap_explanation.data.iloc[0].values, shap_explanation.values[0]):        
        explanation_json = {}

        # > Map the values to strings for interpretability
        if name == "Gender":
            value = "Male" if value == 1 else "Female"
        elif name == "Married":
            value = "Yes" if value == 1 else "No"
        elif name == "Education":
            value = "Not Graduate" if value == 1 else "Graduate"
        elif name == "Self Employed":
            value = "Yes" if value == 1 else "No"

        # > Map "Property Area" to it's original category
        # keep only the value that is equal to 1 
        # since the property area is one-hot encoded
        elif name == "Property Area: Rural":
            if value == 1:
                name = "Property Area"
                value = "Rural"
            else:
                continue
        elif name == "Property Area: Semiurban":
            if value == 1:
                name = "Property Area"
                value = "Semi-urban"
            else:
                continue
        elif name == "Property Area: Urban":
            if value == 1:
                name = "Property Area"
                value = "Urban"
            else:
                continue
        elif name == "Has Credit History":
            value = "Yes" if value == 1 else "No"

        # > Map the "Loan Amount" values to thousands
        # since the original data is in thousands
        elif name == "Loan Amount":
            value = value * 1000

        explanation_json["Name"] = name
        
        # Round numerical features value
        if name in ['Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Loan Term']:
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
    explanation_df.reset_index(drop=True, inplace=True)

    # Rename SHAP value to Impact Score to make it more intuitive
    explanation_df.rename(columns={"SHAP Value": "Impact Score"}, inplace=True)

    return explanation_df

