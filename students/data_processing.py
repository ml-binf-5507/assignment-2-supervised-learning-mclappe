"""
Data loading and preprocessing functions for heart disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(filepath):
    """
    Load the heart disease dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty or malformed
        
    Examples
    --------
    >>> df = load_heart_disease_data('data/heart_disease_uci.csv')
    >>> df.shape
    (270, 15)
    """
    # Hint: Use pd.read_csv()
    # Hint: Check if file exists and raise helpful error if not
    # TODO: Implement data loading
    pass


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, and clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    # TODO: Implement preprocessing
    # - Handle missing values
    # - Encode categorical variables (e.g., sex, cp, fbs, etc.)
    # - Ensure all columns are numeric
    pass


def prepare_regression_data(df, target='chol'):
    """
    Prepare data for linear regression (predicting serum cholesterol).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'chol')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector
    """
    # TODO: Implement regression data preparation
    # - Remove rows with missing chol values
    # - Exclude chol from features
    # - Return X (features) and y (target)
    pass


def prepare_classification_data(df, target='num'):
    """
    Prepare data for classification (predicting heart disease presence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'num')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector (binary)
    """
    # TODO: Implement classification data preparation
    # - Binarize target variable
    # - Exclude target from features
    # - Exclude chol from features
    # - Return X (features) and y (target)
    pass


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        where scaler is the fitted StandardScaler
    """
    # TODO: Implement train/test split and scaling
    # - Use train_test_split with provided parameters
    # - Fit StandardScaler on training data only
    # - Transform both train and test data
    # - Return scaled data and scaler object
    pass
