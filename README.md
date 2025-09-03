# Lab-5
Dimensional reduction 
# Dimensionality Reduction Lab: PCA, LDA, and KPCA

## Overview

This lab is based on Chapter 5 of **Python Machine Learning (Second Edition)** by Sebastian Raschka and Vahid Mirjalili. The chapter covers unsupervised and supervised dimensionality reduction methods, including Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Kernel Principal Component Analysis (KPCA).

In this hands-on lab, you will:
*   Implement PCA from scratch and using scikit-learn.
*   Apply LDA for supervised dimensionality reduction.
*   Use KPCA to handle nonlinear data.
*   Visualize results and analyze the impact of dimensionality reduction on classification tasks.

The lab uses the **Wine dataset** (from UCI) for PCA and LDA, and **synthetic datasets** (half-moons and circles) for KPCA.

## Objectives

*   Understand how PCA maximizes variance for unsupervised dimensionality reduction.
*   Learn how LDA maximizes class separability using label information.
*   Explore KPCA for nonlinear mappings using the RBF kernel.
*   Compare the performance of these techniques on real and synthetic data.
*   Visualize transformed data and evaluate using a simple classifier (e.g., Logistic Regression).

## Prerequisites

*   **Python 3.x**
*   **Libraries**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`
    *   Install via pip: `pip install numpy pandas matplotlib scikit-learn scipy`
*   **Jupyter Notebook** (recommended for interactive execution)

## Setup

1.  Clone this repository: `git clone <repo-url>`
2.  Navigate to the lab directory: `cd lab-dimensionality-reduction`
3.  Launch Jupyter: `jupyter notebook`
4.  Open `dimensionality_reduction_lab.ipynb` (or create one with the code below).
5.  Download the Wine dataset if needed: [wine.data](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)

---

## Part 1: Principal Component Analysis (PCA)

PCA is an unsupervised technique that finds principal components (directions of maximum variance) to reduce dimensionality.

### Step 1.1: Load and Preprocess the Wine Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Wine dataset
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Split into features and labels
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# Standardize features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

```
### Step 1.2: Implement PCA from Scratch
Compute covariance matrix, eigenvalues, and eigenvectors.

```python
import numpy as np

# Compute covariance matrix
cov_mat = np.cov(X_train_std.T)

# Perform eigendecomposition
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Sort eigenvalues in descending order
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# Create projection matrix (top 2 components)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

# Transform training data
X_train_pca = X_train_std.dot(w)
```python
