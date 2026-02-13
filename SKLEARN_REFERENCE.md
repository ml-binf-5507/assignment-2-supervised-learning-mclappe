# Scikit-learn Quick Reference

Essential scikit-learn functions and classes for this assignment.

## Data Preprocessing

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% test, 80% train
    random_state=42,    # Set seed for reproducibility
    stratify=y          # Keep class proportions (for classification)
)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data, transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get fitted scaler for later use
scaler.mean_  # Mean of training data
scaler.scale_  # Standard deviation of training data
```

**Important**: Always fit StandardScaler on training data ONLY, then apply to test data.

## Regression

### ElasticNet

```python
from sklearn.linear_model import ElasticNet

# Create model
model = ElasticNet(
    l1_ratio=0.5,    # Mix of L1 and L2 (0=L2 only, 1=L1 only)
    alpha=0.1,       # Regularization strength
    random_state=42
)

# Fit to training data
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Get R² score
score = model.score(X_test, y_test)  # Returns R² on test set
```

### Grid Search for ElasticNet

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'l1_ratio': [0.3, 0.5, 0.7],
    'alpha': [0.01, 0.1, 1.0]
}

# GridSearchCV does 5-fold cross-validation by default
gs = GridSearchCV(
    ElasticNet(random_state=42),
    param_grid,
    cv=5,           # 5-fold cross-validation
    scoring='r2',   # Metric to optimize
    n_jobs=-1       # Use all CPU cores
)

gs.fit(X_train, y_train)

# Results
gs.best_params_     # Best parameters found
gs.best_score_      # Best cross-validation score
gs.best_estimator_  # Best model (already trained)

# Access all results
results_df = pd.DataFrame(gs.cv_results_)
```

## Classification

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(
    C=1.0,              # Inverse regularization strength (larger = less regularization)
    penalty='l2',       # Type of regularization
    solver='lbfgs',     # Optimization algorithm
    random_state=42,
    max_iter=1000       # Max iterations (increase if not converging)
)

# Fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probability estimates
```

### k-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

# Create model
model = KNeighborsClassifier(
    n_neighbors=5,      # k: number of nearest neighbors
    weights='uniform',  # 'uniform' or 'distance'
    metric='euclidean'  # Distance metric
)

# Fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probability estimates
```

### Grid Search for Classification

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

gs = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',  # Metric: use roc_auc for imbalanced data
    n_jobs=-1
)

gs.fit(X_train, y_train)
gs.best_params_
gs.best_estimator_
```

## Evaluation Metrics

### Regression Metrics

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_pred = model.predict(X_test)

# R² score (higher is better, max = 1.0)
r2 = r2_score(y_test, y_pred)

# Mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Mean absolute error
mae = mean_absolute_error(y_test, y_pred)
```

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ROC and PR metrics (require probability estimates)
auc = roc_auc_score(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

# Get curve data for plotting
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

## Plotting

### ROC Curve

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

y_pred_proba = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP={ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a 2D array of R² scores
heatmap_data = np.array([[0.45, 0.52, 0.49],
                         [0.48, 0.55, 0.51],
                         [0.50, 0.58, 0.54]])

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis',
            xticklabels=['0.01', '0.1', '1.0'],
            yticklabels=['0.3', '0.5', '0.7'])
plt.xlabel('Alpha')
plt.ylabel('L1 Ratio')
plt.title('ElasticNet R² Scores')
plt.show()
```

## Common Patterns

### Full Classification Pipeline

```python
# 1. Load and prepare data
df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Grid search
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
gs = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='roc_auc')
gs.fit(X_train_scaled, y_train)

# 5. Evaluate
best_model = gs.best_estimator_
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
```

## Tips

- **Always scale** before k-NN (distance-based) and Logistic Regression
- **Don't scale** the target variable `y`
- **Fit on training data only** - scale, encoder, etc. should use training data statistics
- **Use `random_state=42`** for reproducibility
- **Use `n_jobs=-1`** in GridSearchCV to use all CPU cores (speeds up computation)
- **Check `model.get_params()`** to see all parameters of a model
