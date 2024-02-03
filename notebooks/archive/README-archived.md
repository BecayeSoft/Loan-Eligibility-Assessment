
# A trust-worthy model for loan eligibility prediction

In this project, I built a machine learning model to assess if an applicant is eligible for a loan or not, then I used GPT-3.5 to generate a report explaining in details the model's decision to a loan's officer. 

> TL;DR: You can find the generated report for a example of loan application in the [reports](reports) folder.

⚠️ This project is still a work in progress. ⚠️

## GUI

The GUI was built using [Streamlit](https://streamlit.io/).

![Alt text](imgs/gui.png)

<!-- 
In this project, I built a Machine Learning model to predict loan eligibility. Since I was working on a problem that involves humans, there were ethical concerns. I needed to make sure that my model was trust-worthy, unbiased and was not discriminating against any group of people. 
Therefore, explanability was at the core of this project. I made sure to explain the model's logic and its predictions using SHAP values. -->

Let me take you through the steps of this project.

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

Based on the data, I came up with the following conditions to reject a loan application:

- If the applicant does not have a credit history, then the loan is rejected.
- If their total income is less than 2500, then the loan is rejected.
- If the loan amount to income ratio is greater than 0.5, then the loan is rejected.

If none of these conditions are met, then the loan is approved.

### Machine Learning

To solve this problem using machine learning, I followed these steps:

- Automated model selection: we will use the [multivariate TPE](https://tech.preferred.jp/en/blog/multivariate-tpe-makes-optuna-even-more-powerful/) to find the combination of model and hyperparameters. [Optuna](https://optuna.org/) is a great library for this.
- Interpretability: we will use [SHAP](https://shap.readthedocs.io/en/latest/) to explain the model's predictions.
- Report generation: we will use [GPT-3.5](https://platform.openai.com/docs/models) to generate a report explaining the model's decision.

## Results

### Condition-based eligibility

When we use the condition-based approach on the training data, we get interesting results:

```
          precision    recall  f1-score   support

    0          0.62      0.49      0.55       148
    1          0.79      0.87      0.83       332

   accuracy                        0.81       480
   macro avg   0.71     0.68       0.69       480
weighted avg   0.74     0.75       0.74       480
```

But this results are hasardous and highly unstable. When we use the same approach on the validation data, we get the following results:

```
          precision    recall  f1-score   support

    0          0.40      0.90      0.56        73
    1          0.98      0.78      0.87       448

   accuracy                        0.80       521
   macro avg   0.69      0.84      0.72       521

![Confusion matrix](reports/archive/multiple-condition-cm.png)

Interestingly, using only one condition (credit history), we get higher scores:

```
          precision    recall  f1-score   support

    0          0.90      0.43      0.58       148
    1          0.79      0.98      0.88       332

   accuracy                        0.81       480
   macro avg   0.85      0.70      0.73       480
weighted avg   0.83      0.81      0.78       480
```

However, this cannot be used in real life as we are approving people only based on their credit history.

![Confusion matrix](reports/archive/single-condition-cm.png)


### Machine Learning

The machine learning model achieved better performance than the (multiple) condition-based approach. 
Interestingly though, the single condition-based approach achieved higher scores.

Machine learning yielded the following results:

```
              precision    recall  f1-score   support

    Rejected       0.40      0.90      0.56        73
    Approved       0.98      0.78      0.87       448

    accuracy                           0.80       521
   macro avg       0.69      0.84      0.72       521
weighted avg       0.90      0.80      0.83       521
```

![Confusion matrix](reports/archive/ml-model-cm.png)

## Conclusion

While the single condition-based approach achieved almost equal f1-score as the machine learning model, we cannot rely on a person's credit history alone to determine if they are eligible for a loan or not. 

On the other hand, the machine learning model is able to capture more complex relationships between the features and the target variable but also generates explanations for its predictions.

### Other approaches

#### Feature engineering

Some experiments were conducted by creating new features from the existing ones:
- Monthly Payment: Monthly payment for the loan
- Total Income: The sum of applicant and co-applicant income
- Amount Income Ratio: The ratio between the monthly payment and the total income
- Amount Income Ratio Percent: The ratio between the monthly payment and the total income, mapped between 0 and 1.

However, these features did not have any impact on the model's performance.


## References

- Dataset: https://www.kaggle.com/datasets/devzohaib/eligibility-prediction-for-loan