import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import load
import shap
from sklearn import set_config
from os.path import dirname, join, abspath, normpath

# Set transformers output to Pandas DataFrame instead of NumPy array
set_config(transform_output="pandas")



global model
global preprocessor

# Current directory
current_dir = dirname(abspath(__file__))

model_path = normpath(join(current_dir, "..", "..", "models", "model.pkl"))
preprocessor_path = normpath(join(current_dir, "..", "..", "models", "preprocessor.pkl"))

# ------- Load preprocessor and model------- #
with open(model_path, 'rb') as f:
    model = load(f)

with open(preprocessor_path, 'rb') as f:
    preprocessor = load(f)


# ------- Generate SHAP explanation ------- #

def generate_shap_explanation(X_test, user_input):
    """
    Generate SHAP explanation for the user input.

    Parameters:
    -----------
    X_test: DataFrame
        Test set
    user_input: DataFrame
        User input
    """
    # ------- Unscaling the test data ------- #
    # Get the scaler and encoder object from the pipeline
    scaler = preprocessor.named_transformers_['numerical']['scaler']
    encoder = preprocessor.named_transformers_['categorical']['onehot']

    # ------- Unscaling the data (user input) ------- #
    data = preprocessor.transform(user_input)

    # Unscale the  data
    data_num_unscaled = scaler.inverse_transform(data[scaler.feature_names_in_])
    data_num_unscaled_df = pd.DataFrame(data=data_num_unscaled, columns=scaler.feature_names_in_)

    # Get the one-hot encoded features
    data_cat = data[encoder.get_feature_names_out()]

    # Reset the index before concatenation
    data_num_unscaled_df.reset_index(drop=True, inplace=True)
    data_cat.reset_index(drop=True, inplace=True)

    # Concat the unscaled numeric data and the categorical data
    data_unscaled = pd.concat([data_num_unscaled_df, data_cat], axis=1)

    # ------- Generate Explanations ------- #
    explainer = shap.LinearExplainer(model, X_test)
    shap_explanation = explainer(data)
    shap_explanation.data = data_unscaled

    return shap_explanation



# ------- Plotting ------- #

def plot_dependence(features, shap_values, data, plot_shape, figsize=(16, 4)):
    """
    Plot the shap dependences for the given features.

    Parameters:
    -----------
    features: list
        List of features to plot
    shap_values: array-like
        Shap values
    data: DataFrame
        Dataframe containing the features
    plot_shape: tuple
        Shape of the plot
    figsize: tuple, default=(16, 4)
        Size of the figure
    """
    _, axs = plt.subplots(ncols=plot_shape[0], nrows=plot_shape[1], figsize=figsize, constrained_layout=True)
    axs = axs.flatten()

    for i, feature in enumerate(features):
        shap.dependence_plot(feature, shap_values, data, ax=axs[i], show=False)

