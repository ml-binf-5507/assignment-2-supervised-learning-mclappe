"""
Assignment 2: Machine Learning for Heart Disease Prediction
============================================================

This package contains functions for:
1. Linear Regression: Predicting cholesterol using ElasticNet
2. Logistic Regression: Classifying heart disease presence
3. k-Nearest Neighbors: Classifying heart disease presence

Students should implement the functions in each module.
"""
# Import submodules to make them available
from . import data_processing
from . import regression
from . import classification
from . import evaluation

__all__ = ['data_processing', 'regression', 'classification', 'evaluation']