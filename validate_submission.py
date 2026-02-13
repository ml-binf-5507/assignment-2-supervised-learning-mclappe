"""
Validation script to verify submissions before pushing.

Run with: python validate_submission.py
"""

import os
import sys
import importlib

def check_files_exist():
    """Check that all required files exist."""
    required_files = [
        'students/__init__.py',
        'students/data_processing.py',
        'students/regression.py',
        'students/classification.py',
        'students/evaluation.py',
        'tests/test_submission.py',
    ]
    
    print("Checking required files...")
    all_exist = True
    for filepath in required_files:
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        all_exist = all_exist and exists
    
    return all_exist


def check_functions_implemented():
    """Check that key functions are implemented (not just pass)."""
    print("\nChecking function implementations...")
    
    sys.path.insert(0, '.')
    
    required_functions = {
        'students.data_processing': [
            'load_heart_disease_data',
            'preprocess_data',
            'prepare_regression_data',
            'prepare_classification_data',
            'split_and_scale',
        ],
        'students.regression': [
            'train_elasticnet_grid',
            'create_r2_heatmap',
            'get_best_elasticnet_model',
        ],
        'students.classification': [
            'train_logistic_regression_grid',
            'train_knn_grid',
            'get_best_logistic_regression',
            'get_best_knn',
        ],
        'students.evaluation': [
            'calculate_r2_score',
            'calculate_classification_metrics',
            'calculate_auroc_score',
            'calculate_auprc_score',
            'generate_auroc_curve',
            'generate_auprc_curve',
        ],
    }
    
    all_implemented = True
    for module_name, functions in required_functions.items():
        try:
            module = importlib.import_module(module_name)
            for func_name in functions:
                if hasattr(module, func_name):
                    print(f"  ✓ {module_name}.{func_name}")
                else:
                    print(f"  ✗ {module_name}.{func_name} - NOT FOUND")
                    all_implemented = False
        except ImportError as e:
            print(f"  ✗ Failed to import {module_name}: {e}")
            all_implemented = False
    
    return all_implemented


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Assignment 2 Submission Validator")
    print("=" * 60)
    
    files_ok = check_files_exist()
    functions_ok = check_functions_implemented()
    
    print("\n" + "=" * 60)
    if files_ok and functions_ok:
        print("✓ All checks passed! Ready to submit.")
        print("=" * 60)
        return 0
    else:
        print("✗ Some checks failed. Please review the issues above.")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
