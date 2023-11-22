import matplotlib.pyplot as plt
import shap


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
