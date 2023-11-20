
# A trust-worthy model for loan eligibility prediction

In this project, I built a Machine Learning model to predict loan eligibility. Since I was working on a problem that involves humans, there were ethical concerns. I needed to make sure that my model was trust-worthy, unbiased and was not discriminating against any group of people. 
Therefore, explanability was at the core of this project. I made sure to explain the model's logic and its predictions using SHAP values.

Let me take you through the steps of this project.

## Situation

Dream Housing Finance company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling the online application form. 

## Analysis

What are the possible solutions to this problem?

### Possible solutions

#### Condition-based eligibility

I could use an if/else to determine if the customer is eligible or not. For example, if they do not have a credit history, they are not eligible.

**Pros**
- Easy to implement

**Cons**
- Not very accurate, does not capture the relationship between the features and the target variable

#### Machine Learning

ML algorithms are more powerful than condition-based eligibility. They can capture complex relationships.

**Pros**
- More accurate
- Smarter

**Cons**
- Unecessary complexity
- Harder to implement
- Explainability

## Action

I first used a condition-based approach before even thinking about machine learning.
I got these poor results:
- Accuracy: 0.81
- F1 score: 0.88

Obviously, I needed to use machine learning. 

