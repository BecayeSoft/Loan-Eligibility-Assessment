from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import DecisionBoundaryDisplay

from xgboost import plot_importance

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["Rejected", "Approved"]
CM_CMAP = "Blues"


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

def print_classification_report(
    model, X, y, title="Classification Report", save=False, save_as=None
):
    """
    Prints the classification report of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The data to evaluate the classification report on.
    y: array-like of shape (n_samples, )
        The true labels.
    """	
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=LABELS)
    print(report)


# ------------------------------------------------------------
# Overall Model Performance Plots
# ------------------------------------------------------------

def plot_model_performance(
    model,
    X,
    y,
    calibration_bins=15,
    figsize=(9, 8)
):
    """
    Plot the confusion matrix, the ROC curve, the precision-recall curve
    and the calibration curve of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The data to evaluate the calibration curve on.
    y: array-like of shape (n_samples, )
        The true labels.
    calibration_bins: int, default=15
        The number of bins to use when plotting the calibration curve.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from src.visualization.model_performance import plot_model_performance
    >>> model = XGBClassifier()
    >>> model.fit(X_train, y_train)
    >>> plot_model_performance(model, X_test, y_test)
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    model_name = model.__class__.__name__

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=axs[0, 0],  cmap=CM_CMAP)
    axs[0, 0].set_title(f"Confusion Matrix")

    # PR-curve
    PrecisionRecallDisplay.from_estimator(
        model, X, y, ax=axs[0, 1], name=model_name, plot_chance_level=True
    )
    axs[0, 1].set_title(f"Precision-Recall Curve")

    # ROC curve
    RocCurveDisplay.from_estimator(
        model, X, y, ax=axs[1, 0], name=model_name, plot_chance_level=True
    )
    axs[1, 0].set_title(f"ROC Curve")

    # Calibration curve
    CalibrationDisplay.from_estimator(
        model, X, y, n_bins=calibration_bins, name=model_name, ax=axs[1, 1]
    )
    axs[1, 1].set_title(f"Calibration Curve")

    fig.suptitle(f"{model_name} Performance", fontsize=16, y=1.03)
    plt.tight_layout()

    plt.show()


# ----------------------------------------------------------------
# 2. Confusion Mattrix, ROC, Precision Recall and Calibration Plots
# -------------------------------------------------------------

def plot_confusion_matrix(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    title1="Set 1",
    title2="Set 2",
    figsize=(5, 4),
):
    """
    Plot the confusion matrix.

    If a second set is provided (X2, y2), plot the confusion matrix of the second set as well.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The first set to evaluate the ROC curve on.
    y1: array-like of shape (n_samples, )
        The true labels of the first confusion matrix curve.
    X2: array-like of shape (n_samples, n_features), optional
        The second set to evaluate the ROC curve on.
    y2: array-like of shape (n_samples, ), optional
        The true labels of the second confusion matrix curve.
    title1: str, default='Set 1'
        The title of the first confusion matrix curve.
    title2: str, default='Set 2'
        The title of the second confusion matrix curve.
    title: str, default='confusion matrix Curves Comparison'
        The title of the plot.
    figsize: tuple, default=(9, 4)
        The size of the figure.
    """
    n_cols = 1
    if X2 is not None and y2 is not None:
        n_cols = 2
    
    fig, axs = plt.subplots(1, n_cols, figsize=figsize)

    # Confusion Matrix 1
    ConfusionMatrixDisplay.from_estimator(model, X1, y1, ax=axs, cmap=CM_CMAP)
    axs.set_title(title1)

    # If a second set is provided, plot the confusion matrix of the second set
    if X2 is not None and y2 is not None:
        ConfusionMatrixDisplay.from_estimator(model, X2, y2, ax=axs[1], cmap=CM_CMAP)
        axs[1].set_title(title2)

    fig.suptitle("Confusion Matrix∙ces", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    title1="Set 1",
    title2="Set 2",
    model_name=None,
    figsize=(5, 4),
    save=False,
    save_as=None,
):
    """
    Plot the ROC curve of the model.

    If a second set is provided (X2, y2), plot the ROC curve of the second set as well.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The first set to evaluate the ROC curve on.
    y1: array-like of shape (n_samples, )
        The true labels of the first ROC curve.
    X2: array-like of shape (n_samples, n_features), optional
        The second set to evaluate the ROC curve on.
    y2: array-like of shape (n_samples, ), optional
        The true labels of the second ROC curve.
    title1: str, default='Set 1'
        The title of the first ROC curve.
    title2: str, default='Set 2'
        The title of the second ROC curve.
    """
    fig, axs = plt.subplots(figsize=figsize)

    model_name = model.__class__.__name__ 

    # If second set is provided
    # Do not plot the chance level for the first set
    plot_chance_first = True if X2 is None and y2 is None else False

    # Plot the ROC curve of the first set
    RocCurveDisplay.from_estimator(
        model, X1, y1, ax=axs, name=title1, alpha=0.8, plot_chance_level=plot_chance_first
    )

    # If the second set is provided, plot the ROC curve of the second set
    if X2 is not None and y2 is not None:
        RocCurveDisplay.from_estimator(
            model, X2, y2, ax=axs, name=title2, alpha=0.8, plot_chance_level=True
        )

    fig.suptitle(f'{model_name} ROC Curve∙s', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    title1="Set 1",
    title2="Set 2",
    figsize=(5, 4),
):
    """
    Plot the precision-recall curve of the model.

    If a second set is provided (X2, y2), plot the precision-recall curve of the second set as well.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The first set to evaluate the ROC curve on.
    y1: array-like of shape (n_samples, )
        The true labels of the first confusion matrix curve.
    X2: array-like of shape (n_samples, n_features), optional
        The second set to evaluate the ROC curve on.
    y2: array-like of shape (n_samples, ), optional
        The true labels of the second confusion matrix curve.
    title1: str, default='Training'
        The title of the first PR curve.
    title2: str, default='Test'
        The title of the second PR curve.
    """
    fig, axs = plt.subplots(figsize=figsize)

    model_name = model.__class__.__name__

    # If second set is provided
    # Do not plot the chance level for the first set
    plot_chance_first = True if X2 is None and y2 is None else False

    # Plot the ROC curve of the first set
    PrecisionRecallDisplay.from_estimator(
        model, X1, y1, ax=axs, name=title1, alpha=0.8, plot_chance_level=plot_chance_first
    )

    # If the second set is provided, plot the ROC curve of the second set
    if X2 is not None and y2 is not None:
        PrecisionRecallDisplay.from_estimator(
            model, X2, y2, ax=axs, name=title2, alpha=0.8, plot_chance_level=True
        )

    fig.suptitle(f'{model_name} Precision-Recall Curve∙s', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()



def plot_calibration_curve(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    calibration_bins=10,
    title1="Set 1",
    title2="Set 2",
    figsize=(5, 4),
):
    """
    Plot the probability calibration curve.
    If a second set is provided (X2, y2), plot the calibration curve of the second set as well.

    The probability calibration curve is a plot of the true probabilities
    against the predicted probabilities.
    It shows us how well the model predicts the true probabilities.

    A perfectly calibrated model should have the predicted probabilities
    approximately equal to the true probabilities. This means that points
    on the calibration curve should lie roughly along the diagonal.

    For more information, see
    https://scikit-learn.org/stable/modules/calibration.html.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The data to evaluate the first calibration curve on.
    y1: array-like of shape (n_samples,)
        The true labels of the first calibration curve.
    X2: array-like of shape (n_samples, n_features), optional
        The data to evaluate the second calibration curve on.
    y2: array-like of shape (n_samples,), optional
        The true labels of the second calibration curve.
    calibration_bins: int, default=10
        The number of bins to use when plotting the calibration curve.
    """
    fig, axs = plt.subplots(figsize=figsize)

    model_name = model.__class__.__name__ 
    
    # Plot the ROC curve of the first set
    CalibrationDisplay.from_estimator(
        model, X1, y1, ax=axs, name=title1, alpha=0.8, n_bins=calibration_bins
    )

    # If the second set is provided, plot the ROC curve of the second set
    if X2 is not None and y2 is not None:
        CalibrationDisplay.from_estimator(
            model, X2, y2, ax=axs, name=title2, alpha=0.8, n_bins=calibration_bins
        )
    
    fig.suptitle(f'{model_name} Calibration Curve', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()
