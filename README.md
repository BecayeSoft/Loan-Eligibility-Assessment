
# A trust-worthy model for loan eligibility prediction

In this project, I built a machine learning model to assess if an applicant is eligible for a loan or not, then I used GPT-3.5 to generate a report explaining in details the model's decision to a loan's officer. 

<!-- 
In this project, I built a Machine Learning model to predict loan eligibility. Since I was working on a problem that involves humans, there were ethical concerns. I needed to make sure that my model was trust-worthy, unbiased and was not discriminating against any group of people. 
Therefore, explanability was at the core of this project. I made sure to explain the model's logic and its predictions using SHAP values. -->

Let me take you through the steps of this project.

> TL;DR: You can find the generated report for a example of loan application in the [reports](reports) folder.

## Situation

Loan officiers manually look at loan applications and decide if the applicant is eligible for a loan or not. This process is time-consuming and prone to human biases. Machine Learning can help automate this process and make it more fair and efficient.

However, since we are dealing with humans, a few questions arise:
- How can we make sure that the model is not discriminating against any group of people?
- How can we make sure that the model is trust-worthy?

Therefore, model's interpretability is at the core of this project.

## Analysis

Do we even need machine learning?

### Machine Learning vs. Condition-based Eligibility

#### Condition-based eligibility

Why not use an if/else solution to determine if the customer is eligible or not. For example, if they do not have a credit history, they are not eligible.

**Pros and cons**:

&nbsp;&nbsp;&nbsp;&nbsp; ✅ Easy to implement \
&nbsp;&nbsp;&nbsp;&nbsp; ❌ Not very accurate since it does not capture the relationship between the features and the target variable

#### Machine Learning

ML algorithms are more powerful than condition-based eligibility as they can capture complex relationships.

**Pros and cons**

&nbsp;&nbsp;&nbsp;&nbsp; ✅ Smarter, more accurate, captures complex relationships \
&nbsp;&nbsp;&nbsp;&nbsp; ❌ Complex, harder to implement, explainability is not straightforward

## Action

Now how do we implement this?

### Condition-based eligibility

Based on the data, I came up with the following conditions.

**Non-eligible applicants**:

- Credit history: if the applicant does not have a credit history, they are not eligible
- Income: if the sum of the total income of the applicant and co-applicant is less than 3,000, they are not eligible

**Eligible applicants**:

To be eligible, the applicant must have:
- Credit history: if the applicant must have a credit history
- Income: if the sum of the total income of the applicant and co-applicant is greater than 3,000 and the have less than 3 dependents
    - Otherwise, the sum of applicant's income and co-applicant's income must be greater than 4,000

### Machine Learning

To solve this problem using machine learning, I followed these steps:

- Automated model selection: we will use the [multivariate TPE](https://tech.preferred.jp/en/blog/multivariate-tpe-makes-optuna-even-more-powerful/) to find the combination of model and hyperparameters. [Optuna](https://optuna.org/) is a great library for this.
- Interpretability: we will use [SHAP](https://shap.readthedocs.io/en/latest/) to explain the model's predictions.
- Report generation: we will use [GPT-3.5](https://platform.openai.com/docs/models) to generate a report explaining the model's decision.

## Results

<!-- I first used a condition-based approach before even thinking about machine learning.
I got the following results:
- Accuracy: 0.81
- F1 score: 0.88
- Precision: 0.88
- Recall: 0.88

Obviously, I needed to use machine learning.  -->


## References

- Dataset: