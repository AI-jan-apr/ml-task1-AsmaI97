# Breast Cancer — Binary Classification

## Overview

This project builds and compares multiple binary classification models to predict whether a breast tumor is **malignant (0)** or **benign (1)** using the Breast Cancer Wisconsin Dataset from `scikit-learn`.

The experiment explores the effect of different **iteration counts** on Logistic Regression and different **K values** on KNN, alongside a baseline SVM model.

---

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numerical features (radius, texture, area, smoothness, etc.)
- **Target:** Binary (0 = Malignant, 1 = Benign)
- **Missing values:** None

> ⚠️ Feature scaling is NOT applied in this task.

---

## Train-Test Split

| Parameter | Value |
|---|---|
| Test size | 20% |
| Random state | 42 |
| Stratify | Yes |

---

## Models Trained

### Logistic Regression
Three variants were trained to study the effect of the number of iterations on convergence and performance:
- **Default** (max_iter=100)
- **1,000 iterations** (max_iter=1000)
- **10,000 iterations** (max_iter=10000)

### Support Vector Machine (SVM)
- Default parameters with RBF kernel

### K-Nearest Neighbors (KNN)
Three variants were trained to study the effect of K on performance:
- **k=3**
- **k=5**
- **k=7**

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression (default) | 0.9561 | 0.9467 | 0.9861 | 0.9660 |
| Logistic Regression (1,000 iter) | 0.9561 | 0.9589 | 0.9722 | 0.9655 |
| Logistic Regression (10,000 iter) | **0.9649** | **0.9595** | **0.9861** | **0.9726** |
| SVM | 0.9298 | 0.9211 | 0.9722 | 0.9459 |
| KNN (k=3) | 0.9298 | 0.9444 | 0.9444 | 0.9444 |
| KNN (k=5) | 0.9123 | 0.9429 | 0.9167 | 0.9296 |
| KNN (k=7) | 0.9298 | 0.9444 | 0.9444 | 0.9444 |

> **Bold** = best score per metric.

---

## Key Findings

### Best Performing Model
**Logistic Regression with 10,000 iterations** achieved the best results across all metrics (Accuracy: 0.9649, F1: 0.9726). This shows that giving the solver enough iterations to fully converge on unscaled data makes a meaningful difference.

### Effect of Iterations on Logistic Regression
Increasing iterations from default (100) to 10,000 improved accuracy and F1-score, confirming that the default solver does not fully converge when features are unscaled. Beyond a certain point however, additional iterations yield diminishing returns.

### Effect of K on KNN
Among KNN variants, **k=3** outperformed k=5 on most metrics. Smaller K values capture more local patterns in this dataset, though they are generally more sensitive to noise. KNN overall ranked lowest among all models, which is expected since it is highly sensitive to unscaled feature magnitudes.

### Most Important Metric in a Medical Context
**Recall (Sensitivity)** is the most critical metric in breast cancer diagnosis. A **false negative** — predicting a tumor as benign when it is actually malignant — means a cancer goes undetected, which can be life-threatening. A **false positive** leads to additional tests, which is undesirable but far less dangerous. All Logistic Regression variants achieved a Recall of 0.9861, making them the safest choice in this medical context.

---

## Project Structure

```
breast-cancer-binary-classification/
├── Classification_models.ipynb   # Full training, evaluation, and comparison notebook
└── README.md        # Project documentation
```

---

## How to Run

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn jinja2
   ```
3. Open and run `Classification_models.ipynb` in Jupyter or VS Code.
