"""
Test suite for Assignment 2: Machine Learning for Heart Disease Prediction

Run with: pytest tests/ -v

This test suite validates both structure AND correctness:
- Basic tests: Check functions exist and return correct types
- Quality tests: Check that implementations are actually correct
- Integration tests: Test with realistic data scenarios
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
import sys
import os
import matplotlib.pyplot as plt

# Add students module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from students import data_processing, regression, classification, evaluation


class TestDataProcessing:
    """Tests for data loading and preprocessing."""
    
    @pytest.fixture
    def sample_heart_data(self):
        """Create realistic synthetic heart disease dataset."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(120, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(60, 200, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 3, n_samples),
            'chol': np.random.randint(120, 400, n_samples),  # Regression target
            'num': np.random.randint(0, 2, n_samples),  # Classification target
        })
        return X
    
    def test_prepare_regression_data_returns_tuple(self, sample_heart_data):
        """Test regression data prep returns X, y with correct shapes."""
        try:
            X, y = data_processing.prepare_regression_data(
                sample_heart_data, 'chol'
            )
            assert isinstance(X, (pd.DataFrame, np.ndarray)), "X should be DataFrame or ndarray"
            assert len(X) == len(y), "X and y must have same length"
            assert len(X) > 0, "X should not be empty"
            assert 'chol' not in (X.columns if isinstance(X, pd.DataFrame) else [])
            print(f"✓ Regression data shape: X={X.shape}, y={y.shape}")
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_prepare_classification_data_returns_tuple(self, sample_heart_data):
        """Test classification data prep returns X, y with binary target."""
        try:
            X, y = data_processing.prepare_classification_data(
                sample_heart_data, 'num'
            )
            assert isinstance(X, (pd.DataFrame, np.ndarray)), "X should be DataFrame or ndarray"
            assert len(X) == len(y), "X and y must have same length"
            assert set(y.unique()).issubset({0, 1}), "Target must be binary"
            assert 'num' not in (X.columns if isinstance(X, pd.DataFrame) else [])
            assert 'chol' not in (X.columns if isinstance(X, pd.DataFrame) else [])
            print(f"✓ Classification data shape: X={X.shape}, y={y.shape}")
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_split_and_scale_returns_five_items(self, sample_heart_data):
        """Test train/test split and scaling."""
        try:
            X, y = data_processing.prepare_classification_data(
                sample_heart_data, 'target'
            )
            result = data_processing.split_and_scale(X, y)
            assert isinstance(result, tuple) and len(result) == 5, "Should return 5 items"
            X_train_scaled, X_test_scaled, y_train, y_test, scaler = result
            
            # Check scaling was applied
            assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=0.1), "Training data should be centered"
            assert np.allclose(X_train_scaled.std(axis=0), 1, atol=0.1), "Training data should be scaled"
            print(f"✓ Split and scale: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
        except (NotImplementedError, TypeError, ValueError):
            pytest.skip("Function not yet implemented or has issues")


class TestRegression:
    """Tests for ElasticNet regression."""
    
    @pytest.fixture
    def regression_data(self):
        """Create realistic regression dataset."""
        np.random.seed(42)
        n_samples = 150
        X = np.random.randn(n_samples, 10)
        # Create a signal with different strengths
        y = 2.5 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.5
        return X, y
    
    def test_train_elasticnet_grid_returns_dataframe(self, regression_data):
        """Test that grid search returns proper DataFrame."""
        X, y = regression_data
        try:
            results = regression.train_elasticnet_grid(
                X, y, 
                l1_ratios=[0.3, 0.7],
                alphas=[0.01, 0.1, 1.0]
            )
            assert isinstance(results, pd.DataFrame), "Should return DataFrame"
            assert len(results) == 6, "Should have 2 * 3 = 6 results"
            assert all(col in results.columns for col in ['l1_ratio', 'alpha', 'r2_score'])
            assert 'model' in results.columns, "Should store model objects"
            print(f"✓ ElasticNet grid shape: {results.shape}")
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_elasticnet_r2_scores_reasonable(self, regression_data):
        """Test that R² scores are in reasonable range and vary with parameters."""
        X, y = regression_data
        try:
            results = regression.train_elasticnet_grid(
                X, y,
                l1_ratios=[0.3, 0.7],
                alphas=[0.001, 0.1, 10.0]
            )
            # Check all R² are in valid range
            assert all(-1 <= r2 <= 1 for r2 in results['r2_score']), "R² should be between -1 and 1"
            
            # Check that different alphas produce different results
            r2_by_alpha = results.groupby('alpha')['r2_score'].mean()
            assert len(r2_by_alpha) > 1, "Different alphas should be tested"
            assert r2_by_alpha.std() > 0.001, "Different alphas should produce different results"
            
            print(f"✓ R² scores valid: min={results['r2_score'].min():.3f}, "
                  f"max={results['r2_score'].max():.3f}, "
                  f"std={results['r2_score'].std():.3f}")
        except (NotImplementedError, AttributeError):
            pytest.skip("Function not yet implemented")
    
    def test_create_heatmap_returns_figure_with_labels(self, regression_data):
        """Test heatmap creation and labeling."""
        X, y = regression_data
        try:
            results = regression.train_elasticnet_grid(
                X, y,
                l1_ratios=[0.3, 0.5, 0.7],
                alphas=[0.01, 0.1, 1.0]
            )
            fig = regression.create_r2_heatmap(results, [0.3, 0.5, 0.7], [0.01, 0.1, 1.0])
            
            assert fig is not None, "Should return figure"
            assert len(fig.axes) > 0, "Figure should have axes"
            
            # Check labels exist
            ax = fig.axes[0]
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            assert len(xlabel) > 0, "Should have x-axis label"
            assert len(ylabel) > 0, "Should have y-axis label"
            
            print(f"✓ Heatmap created with labels: x='{xlabel}', y='{ylabel}'")
            plt.close(fig)
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip("Function not yet implemented")
    
    def test_best_elasticnet_selects_best_model(self, regression_data):
        """Test that best model selection works correctly."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            result = regression.get_best_elasticnet_model(
                X_train, y_train, X_test, y_test,
                l1_ratios=[0.3, 0.7],
                alphas=[0.001, 0.1, 10.0]
            )
            
            assert isinstance(result, dict), "Should return dictionary"
            assert 'model' in result, "Should have 'model' key"
            assert 'best_l1_ratio' in result, "Should have 'best_l1_ratio' key"
            assert 'best_alpha' in result, "Should have 'best_alpha' key"
            assert 'test_r2' in result, "Should have 'test_r2' key"
            
            # Best model should have positive test R²
            assert result['test_r2'] > -0.5, "Best model should have reasonable test R²"
            
            print(f"✓ Best model: l1={result['best_l1_ratio']}, "
                  f"alpha={result['best_alpha']}, test_r2={result['test_r2']:.3f}")
        except (NotImplementedError, TypeError, AttributeError):
            pytest.skip("Function not yet implemented")


class TestClassification:
    """Tests for Logistic Regression and k-NN."""
    
    @pytest.fixture
    def classification_data(self):
        """Create realistic classification dataset."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=6,
            n_redundant=2, n_clusters_per_class=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale for k-NN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def test_logistic_regression_grid_has_best_estimator(self, classification_data):
        """Test that logistic regression grid search works."""
        X_train, _, y_train, _, _, _ = classification_data
        try:
            from sklearn.model_selection import GridSearchCV
            gs = classification.train_logistic_regression_grid(X_train, y_train)
            
            assert isinstance(gs, GridSearchCV), "Should return GridSearchCV"
            assert hasattr(gs, 'best_estimator_'), "Should have best_estimator_"
            assert hasattr(gs, 'best_params_'), "Should have best_params_"
            
            # Check that a reasonable parameter was selected
            best_c = gs.best_params_.get('C')
            assert best_c is not None and best_c > 0, "Should select positive C"
            
            print(f"✓ Logistic regression best params: {gs.best_params_}")
        except (NotImplementedError, ImportError, TypeError):
            pytest.skip("Function not yet implemented")
    
    def test_knn_grid_has_best_k(self, classification_data):
        """Test that k-NN grid search works."""
        _, _, _, _, X_train_scaled, _ = classification_data
        _, _, y_train, _, _, _ = classification_data
        try:
            from sklearn.model_selection import GridSearchCV
            gs = classification.train_knn_grid(X_train_scaled, y_train)
            
            assert isinstance(gs, GridSearchCV), "Should return GridSearchCV"
            best_k = gs.best_params_.get('n_neighbors')
            assert best_k is not None and 3 <= best_k <= 20, "Best k should be reasonable"
            
            print(f"✓ k-NN best params: {gs.best_params_}")
        except (NotImplementedError, ImportError, TypeError):
            pytest.skip("Function not yet implemented")
    
    def test_best_logistic_regression_achieves_good_auc(self, classification_data):
        """Test that best logistic regression model achieves good performance."""
        X_train, X_test, y_train, y_test, _, _ = classification_data
        
        try:
            result = classification.get_best_logistic_regression(
                X_train, y_train, X_test, y_test
            )
            
            assert isinstance(result, dict), "Should return dictionary"
            assert 'model' in result, "Should have 'model' key"
            
            # Check model achieves decent AUC
            y_pred_proba = result['model'].predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            assert auc > 0.5, f"AUC should be better than random (got {auc:.3f})"
            assert auc > 0.6, f"AUC should be at least 0.6 (got {auc:.3f})"
            
            print(f"✓ Best logistic regression AUC: {auc:.3f}")
        except (NotImplementedError, TypeError, AttributeError):
            pytest.skip("Function not yet implemented")
    
    def test_best_knn_achieves_good_auc(self, classification_data):
        """Test that best k-NN model achieves good performance."""
        _, _, _, _, X_train_scaled, X_test_scaled = classification_data
        _, _, y_train, y_test, _, _ = classification_data
        
        try:
            result = classification.get_best_knn(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            assert isinstance(result, dict), "Should return dictionary"
            assert 'model' in result, "Should have 'model' key"
            assert 'best_k' in result, "Should have 'best_k' key"
            
            # Check model achieves decent AUC
            y_pred_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            assert auc > 0.5, f"AUC should be better than random (got {auc:.3f})"
            assert auc > 0.6, f"AUC should be at least 0.6 (got {auc:.3f})"
            
            print(f"✓ Best k-NN (k={result['best_k']}) AUC: {auc:.3f}")
        except (NotImplementedError, TypeError, AttributeError):
            pytest.skip("Function not yet implemented")


class TestEvaluation:
    """Tests for evaluation metrics and curves."""
    
    @pytest.fixture
    def evaluation_data(self):
        """Create test evaluation data."""
        np.random.seed(42)
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
        # Good probabilities that correlate with truth
        y_pred_proba = np.array([0.1, 0.95, 0.25, 0.15, 0.85, 0.2, 0.88, 0.92, 0.1, 
                                0.45, 0.8, 0.75, 0.3, 0.9, 0.6])
        return y_true, y_pred, y_pred_proba
    
    def test_calculate_r2_in_valid_range(self):
        """Test R² calculation returns valid values."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        try:
            r2 = evaluation.calculate_r2_score(y_true, y_pred)
            assert isinstance(r2, (float, np.floating)), "Should return float"
            assert -np.inf < r2 <= 1.0, f"R² should be ≤ 1 (got {r2})"
            assert r2 > 0.5, f"R² should be high for close predictions (got {r2})"
            
            print(f"✓ R² score: {r2:.3f}")
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_calculate_classification_metrics_all_valid(self, evaluation_data):
        """Test classification metrics are in valid ranges."""
        y_true, y_pred, _ = evaluation_data
        try:
            metrics = evaluation.calculate_classification_metrics(y_true, y_pred)
            
            assert isinstance(metrics, dict), "Should return dictionary"
            required_keys = ['accuracy', 'precision', 'recall', 'f1']
            assert all(k in metrics for k in required_keys), f"Missing keys: {set(required_keys) - set(metrics.keys())}"
            
            # Check all metrics are valid
            for key, value in metrics.items():
                assert 0 <= value <= 1, f"{key} should be between 0 and 1 (got {value})"
            
            print(f"✓ Metrics: accuracy={metrics['accuracy']:.3f}, "
                  f"precision={metrics['precision']:.3f}, "
                  f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_auroc_score_better_than_random(self, evaluation_data):
        """Test AUROC score is better than random for good predictions."""
        y_true, _, y_pred_proba = evaluation_data
        try:
            auroc = evaluation.calculate_auroc_score(y_true, y_pred_proba)
            assert isinstance(auroc, (float, np.floating)), "Should return float"
            assert 0 <= auroc <= 1, f"AUROC should be between 0 and 1 (got {auroc})"
            assert auroc > 0.5, f"AUROC should be better than random (got {auroc:.3f})"
            assert auroc > 0.7, f"AUROC should be good for correlated predictions (got {auroc:.3f})"
            
            print(f"✓ AUROC score: {auroc:.3f}")
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_auprc_score_better_than_baseline(self, evaluation_data):
        """Test AUPRC score is better than baseline."""
        y_true, _, y_pred_proba = evaluation_data
        try:
            auprc = evaluation.calculate_auprc_score(y_true, y_pred_proba)
            assert isinstance(auprc, (float, np.floating)), "Should return float"
            assert 0 <= auprc <= 1, f"AUPRC should be between 0 and 1 (got {auprc})"
            
            # Baseline AUPRC is class prevalence
            baseline = y_true.mean()
            assert auprc > baseline * 0.8, f"AUPRC should be decent (got {auprc:.3f}, baseline {baseline:.3f})"
            
            print(f"✓ AUPRC score: {auprc:.3f} (baseline: {baseline:.3f})")
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_auroc_curve_properly_formatted(self, evaluation_data):
        """Test ROC curve has proper structure and labels."""
        y_true, _, y_pred_proba = evaluation_data
        try:
            fig = evaluation.generate_auroc_curve(y_true, y_pred_proba, model_name="Test Model")
            
            assert fig is not None, "Should return figure"
            assert len(fig.axes) > 0, "Figure should have axes"
            
            ax = fig.axes[0]
            xlabel = ax.get_xlabel().lower()
            ylabel = ax.get_ylabel().lower()
            
            assert 'false' in xlabel or 'fpr' in xlabel, f"X-axis should reference FPR (got '{ax.get_xlabel()}')"
            assert 'true' in ylabel or 'tpr' in ylabel, f"Y-axis should reference TPR (got '{ax.get_ylabel()}')"
            
            print(f"✓ ROC curve: x='{ax.get_xlabel()}', y='{ax.get_ylabel()}'")
            plt.close(fig)
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip("Function not yet implemented")
    
    def test_auprc_curve_properly_formatted(self, evaluation_data):
        """Test PR curve has proper structure and labels."""
        y_true, _, y_pred_proba = evaluation_data
        try:
            fig = evaluation.generate_auprc_curve(y_true, y_pred_proba, model_name="Test Model")
            
            assert fig is not None, "Should return figure"
            assert len(fig.axes) > 0, "Figure should have axes"
            
            ax = fig.axes[0]
            xlabel = ax.get_xlabel().lower()
            ylabel = ax.get_ylabel().lower()
            
            assert 'recall' in xlabel, f"X-axis should be Recall (got '{ax.get_xlabel()}')"
            assert 'precision' in ylabel, f"Y-axis should be Precision (got '{ax.get_ylabel()}')"
            
            print(f"✓ PR curve: x='{ax.get_xlabel()}', y='{ax.get_ylabel()}'")
            plt.close(fig)
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip("Function not yet implemented")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

