# EDA
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Set transformers output to Pandas DataFrame instead of NumPy array
from sklearn import set_config
set_config(transform_output="pandas")


def create_preprocessing_pipeline(num_features, cat_features):
    """
    Create preprocessor pipeline for numeric and categorical data
    
    Parameters:
    -----------
    num_features: list
        List of numeric features
    cat_features: list
        List of categorical features

    Returns:
    --------
    preprocessor: ColumnTransformer
        Preprocessor pipeline object
    """
    # Preprocessing for numerical data
    numeric_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', PowerTransformer())
    ])

    # Preprocessing for categorical data
    categoric_pipe = Pipeline([
        ( 'imputer', SimpleImputer(strategy='most_frequent') ),
        ( 'onehot', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore') )
    ])

    # Combining the preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipe, num_features),
            ('categorical', categoric_pipe, cat_features),
        ], 
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    return preprocessor


def train_val_split(df, val_size=0.2):
    """
    Perform a stratified split of the data into train and validation set
    
    Parameters:
    -----------
    df: DataFrame
        The DataFrame to split

    Returns:
    --------
    X_train: array-like
        Train features
    X_test: array-like
        Validation features
    y_train: array-like
        Train target
    y_test: array-like
        Validation target
    """
    train_df, val_df = train_test_split(
        df, stratify=df["Loan_Status"], test_size=val_size, random_state=42
    )

    return train_df, val_df
