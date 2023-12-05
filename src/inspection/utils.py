from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt
import numpy as np



def plot_learning_curve(
    model, X, y, scoring="roc_auc", title="Learning Curve", save=False, save_as=None
):
    """
    Plots the learning curve of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The training input samples.
    y: array-like of shape (n_samples, )
        The target values.
    scoring: str, default='f1'
        The scoring metric to use.
        E.g. 'roc_auc', 'f1', 'accuracy', 'precision', 'recall'.
    """
    LearningCurveDisplay.from_estimator(model, X, y, scoring=scoring, n_jobs=-1)
    plt.title("Learning Curve", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()


def plot_validation_curves(
    model,
    X,
    y,
    params,
    scoring="roc_auc",
    title="Validation Curve",
    figsize=(9, 8),
):
    """
    Plots the validation curves of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The training input samples.
    y: array-like of shape (n_samples, )
        The target values.
    params: list of dicts
        A list of dictionaries containing the parameter name and the range of values to be tested.
    scoring: str, default='roc_auc'
        The scoring metric to use.
        E.g. 'roc_auc', 'f1', 'accuracy', 'precision', 'recall'.
    figsize: tuple of shape (width, height), default=(9, 8)
        The size of the figure.
    """
    # Calculate the number of axes
    n_params = len(params)
    n_rows = np.ceil(np.sqrt(n_params)).astype(int)
    n_cols = np.floor(np.sqrt(n_params)).astype(int)

    _, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    for i, param in enumerate(params):
        name = param["name"]
        range = param["range"]
        ValidationCurveDisplay.from_estimator(
            model,
            X,
            y,
            param_name=name,
            param_range=range,
            scoring=scoring,
            ax=axs[i],
            n_jobs=-1,
        )

    plt.suptitle("Validation Curve", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()
