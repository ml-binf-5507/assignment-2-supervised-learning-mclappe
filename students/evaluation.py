"""
Model evaluation functions: metrics and ROC/PR curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, auc as compute_auc, r2_score
)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True target values
    y_pred : np.ndarray or pd.Series
        Predicted target values
        
    Returns
    -------
    float
        R² score (between -inf and 1, higher is better)
    """
    # TODO: Implement R² calculation
    # Use sklearn's r2_score

    return r2_score(y_true, y_pred)


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred : np.ndarray or pd.Series
        Predicted binary labels
        
    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # TODO: Implement metrics calculation
    # Return dictionary with all four metrics

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_auroc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the ROC Curve (AUROC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUROC score (between 0 and 1)
    """
    # TODO: Implement AUROC calculation
    # Use sklearn's roc_auc_score
    return roc_auc_score(y_true, y_pred_proba)


def calculate_auprc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUPRC score (between 0 and 1)
    """
    # TODO: Implement AUPRC calculation
    # Use sklearn's average_precision_score
    return average_precision_score(y_true, y_pred_proba)


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None):
    """
    Generate and plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement ROC curve plotting
    # - Calculate ROC curve using roc_curve()
    # - Calculate AUROC using auc()
    # - Plot curve with label showing AUROC score
    # - Add diagonal reference line
    # - Set labels: "False Positive Rate", "True Positive Rate"
    # - Save to output_path if provided
    # - Return figure and/or axes
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = auc(fpr, tpr)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, label=f'ROC Curve ({model_name} AUROC={auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve ({model_name})')
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path:
        fig.savefig(output_path)

    return fig

def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None):
    """
    Generate and plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement PR curve plotting
    # - Calculate precision-recall curve using precision_recall_curve()
    # - Calculate AUPRC using average_precision_score()
    # - Plot curve with label showing AUPRC score
    # - Add horizontal baseline (prevalence)
    # - Set labels: "Recall", "Precision"
    # - Save to output_path if provided
    # - Return figure and/or axes
    
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)


    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(recall_vals, precision_vals, label=f'PR Curve ({model_name} AUPRC={auprc:.3f})')

    # Baseline = prevalence of positive class
    prevalence = np.mean(y_true)
    ax.axhline(y=prevalence, color='k', linestyle='--', label='Prevalence')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve ({model_name})')
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba_log : np.ndarray or pd.Series
        Predicted probabilities from logistic regression
    y_pred_proba_knn : np.ndarray or pd.Series
        Predicted probabilities from k-NN
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2 subplots (ROC and PR curves)
    """
    # TODO: Implement comparison plotting
    # - Create figure with 1x2 subplots
    # - Left: ROC curves for both models
    # - Right: PR curves for both models
    # - Add legends with AUROC/AUPRC scores
    # - Save to output_path if provided
    # - Return figure
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    generate_auroc_curve(
        y_true, 
        y_pred_proba_log, 
        model_name="Logistic Regression",
        ax=axes[0])
    generate_auroc_curve(
        y_true, 
        y_pred_proba_knn, 
        model_name="k-NN",
        ax=axes[0])
    axes[0].set_title("ROC Curve Comparison")

    generate_auprc_curve(
        y_true, 
        y_pred_proba_log, 
        model_name="Logistic Regression",
        ax=axes[1])
    generate_auprc_curve(
        y_true, 
        y_pred_proba_knn, 
        model_name="k-NN",
        ax=axes[1])
    axes[1].set_title("Precision-Recall Curve Comparison")

    if output_path:
        fig.savefig(output_path)

    return fig
