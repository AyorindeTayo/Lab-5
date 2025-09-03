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
```
### Step 1.3: Visualize Explained Variance
```python
import matplotlib.pyplot as plt

# Calculate explained variance ratios
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Plot
plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

### Step 1.4: PCA with scikit-learn
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca_sk = pca.fit_transform(X_train_std)
X_test_pca_sk = pca.transform(X_test_std)
```

### Step 1.5: Classification and Visualization
Use Logistic Regression on reduced data and visualize the decision regions.
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr.fit(X_train_pca_sk, y_train)

# Helper function to plot decision regions (from the book)
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='best')

# Plot decision regions for training set
plot_decision_regions(X_train_pca_sk, y_train, classifier=lr)
plt.title('Logistic Regression on PCA-transformed Data (Training)')
plt.show()

# Check accuracy on test set
print('Test Accuracy: %.3f' % lr.score(X_test_pca_sk, y_test))
```
## Part 2: Linear Discriminant Analysis (LDA)
