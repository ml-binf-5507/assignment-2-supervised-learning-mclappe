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
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} was not found.")
    except ValueError:
        raise ValueError(f"The file {filepath} is not a valid CSV, either empty or malformed.")

    return df


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

    categorical_cols = [
        "sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    numeric_cols = df.columns.difference(categorical_cols)

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
                                               
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


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

    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])

    return X, y


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

    df = df.dropna(subset=[target])

    y = (df[target] > 0).astype(int)

    drop_cols = [target]
    if 'chol' in df.columns:
        drop_cols.append('chol')

    X = df.drop(columns=drop_cols)

    return X, y


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
