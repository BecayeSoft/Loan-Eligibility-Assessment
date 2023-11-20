# EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Training
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, average_precision_score

import optuna

# Set transformers output to Pandas DataFrame instead of NumPy array
from sklearn import set_config
set_config(transform_output="pandas")



def tune_params(
        X_train, y_train, X_val, y_val, n_trials=200
    ):
    """
    Maximize the objective function
    """

    # ------- Objective function ------- #
    def objective(trial):
        """
        Objective function for hyperparameter tuning

        Parameters:
        -----------
        X_train: DataFrame
            Training data
        y_train: Series
            Training labels
        X_val: DataFrame
            Validation data
        y_val: Series
            Validation labels

        Returns:
        --------
        f1: float
            The f1 score on the validation set
        """

        clf = trial.suggest_categorical(
            'classifier', 
            ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost', 'MLP']
        )

        if clf == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            }
            model = LogisticRegression(**params)

        elif clf == 'SVM':
            params = {
                'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            }
            # gamma is not supported for linear kernel
            if params['kernel'] != 'linear':
                params['gamma']: trial.suggest_categorical('gamma', ['scale', 'auto'])


            model = SVC(**params)

        elif clf == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_float('min_samples_split', 0.8, 1.0),
                'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
            }
            model = RandomForestClassifier(**params)

        elif clf == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, log=True)
            }
            model = XGBClassifier(**params)

        elif clf == 'MLP':
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100, 50), (100, 100)]),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-2, log=True),
            }
            model = MLPClassifier(**params)

        elif clf == 'NaiveBayes':
            model = GaussianNB()
            params = {}

        # Train and evaluate the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        return f1


    # ------- Optimization ------- #
    # multivariate TPE takes hyperparameters dependencies into consideration
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)


    return study.best_params