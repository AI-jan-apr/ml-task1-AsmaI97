[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/3TS0mELD)
# Machine Learning Task 1
## Breast Cancer — Binary Classification

---

## Objective

In this task, you will build and compare multiple **binary classification** models to predict whether a tumor is:

- **0 — Malignant (Cancerous)**
- **1 — Benign (Non-cancerous)**

You must use the following models covered in class:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The focus of this task is **model training, evaluation, and comparison**.

⚠️ Feature scaling is NOT allowed in this task.

---

## Dataset

We will use the **Breast Cancer Wisconsin Dataset**, available directly in `scikit-learn`.

### Dataset Overview

- 569 samples
- 30 numerical features
- Binary target variable
- No missing values

Each feature represents a measurement extracted from a digitized image of a breast mass (e.g., radius, texture, area, smoothness, concavity, symmetry, etc.).

---

## Dataset Loading

Use the following code to load the dataset:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
```

---

## Required Tasks

### 1. Train-Test Split

Split the dataset using:

- `test_size = 0.2`
- `random_state = 42`
- `stratify = y`

### 2. Model Training

Train the following models:

- Logistic Regression
- SVM
- KNN

Use default parameters unless clearly justified.

### 3. Model Evaluation

For each model, compute:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 4. Model Comparison

Create a comparison table summarizing the evaluation metrics for all models.

Then write a short conclusion answering:

- Which model performed best?
- In a medical context, which metric is most important and why?

---

## Project Structure

Your project must follow this structure:

```
breast-cancer-binary-classification/
├── modeling.ipynb
└── README.md
```

---

## Submission Requirements

- Clean and organized notebook
- Clear metric comparison
- Written conclusion
