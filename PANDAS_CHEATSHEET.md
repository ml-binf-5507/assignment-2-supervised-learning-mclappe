# Pandas Cheatsheet for Assignment 2

Quick reference for common pandas operations used in this assignment.

## Loading Data

```python
import pandas as pd

# Load CSV file
df = pd.read_csv('data/heart.csv')

# Show first few rows
df.head()

# Get info about the dataset
df.info()

# Get summary statistics
df.describe()
```

## Inspecting Data

```python
# Shape: (n_rows, n_columns)
df.shape

# Column names
df.columns

# Data types
df.dtypes

# Check for missing values
df.isnull().sum()

# Get unique values in a column
df['column_name'].unique()
```

## Selecting Data

```python
# Select single column (returns Series)
df['column_name']

# Select multiple columns (returns DataFrame)
df[['col1', 'col2', 'col3']]

# Select by row index
df.iloc[0]  # First row
df.iloc[0:5]  # First 5 rows

# Select rows where condition is true
df[df['age'] > 50]
df[(df['age'] > 50) & (df['sex'] == 1)]
```

## Cleaning Data

```python
# Drop rows with missing values
df.dropna()

# Drop specific column
df.drop('column_name', axis=1)

# Remove rows where target is NaN
df = df.dropna(subset=['target'])

# Replace values
df['column'].replace(0, 1)
```

## Creating New Columns

```python
# Create new column from existing
df['new_col'] = df['col1'] + df['col2']

# Create column based on condition
df['age_group'] = df['age'].apply(lambda x: 'young' if x < 50 else 'old')
```

## Handling Categorical Data

```python
# Get numerical representation of categories
df['category'].astype('category').cat.codes

# One-hot encoding
pd.get_dummies(df['column'], prefix='prefix')

# Label encoding (0, 1, 2, ...)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['original'])
```

## Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% test, 80% train
    random_state=42,        # For reproducibility
    stratify=y              # Keep class balance
)
```

## Common Operations

```python
# Convert to numpy array
arr = df.values
arr = df['column'].values

# Get column as list
lst = df['column'].tolist()

# Rename columns
df.rename(columns={'old_name': 'new_name'})

# Merge dataframes
df_merged = df1.merge(df2, on='key_column')

# Get rows where value is in list
df[df['column'].isin([1, 2, 3])]

# Sort by column
df.sort_values('column', ascending=False)
```

## Debugging Tips

```python
# Print first few rows
print(df.head())

# Check data types and missing values
print(df.info())

# Look at a specific slice
print(df[['col1', 'col2']].head(10))

# Count values
print(df['column'].value_counts())
```

## Common Mistakes to Avoid

❌ **Wrong**: `df['col1', 'col2']` → Use `df[['col1', 'col2']]` (double brackets)

❌ **Wrong**: `df.drop('col1')` → Use `df.drop('col1', axis=1)` (need to specify axis)

❌ **Wrong**: Modifying original data without assignment → Use `df = df.drop(...)` or `df.drop(..., inplace=True)`

❌ **Wrong**: Forgetting to use `[['col']]` when you need a DataFrame (not Series)

✓ **Right**: `X = df.drop('target', axis=1)` → Keeps X as DataFrame
