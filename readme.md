

# 🧠 Stroke Prediction – Machine Learning Pipeline

> End-to-end machine learning workflow for predicting stroke risk using structured health data.

---

## 📌 Project Overview

This project implements a complete supervised machine learning pipeline for **binary classification (Stroke / No Stroke)**.

The notebook includes:

* Data inspection and preprocessing
* Exploratory Data Analysis (EDA)
* Outlier detection
* Missing value treatment
* Categorical encoding
* Feature selection
* Handling class imbalance (SMOTE)
* Model training (multiple algorithms)
* Hyperparameter tuning (Optuna)
* Model persistence (Pickle)

The final optimized model is saved for reuse and deployment.

---

## 📂 Project Structure

```
├── code.ipynb
├── stroke-data
├── README.md
```

---

## 🧾 Dataset Description

The dataset contains patient-level medical and demographic information:

| Feature           | Description             |
| ----------------- | ----------------------- |
| age               | Patient age             |
| gender            | Male / Female           |
| hypertension      | Binary indicator        |
| heart_disease     | Binary indicator        |
| ever_married      | Yes / No                |
| work_type         | Employment category     |
| Residence_type    | Urban / Rural           |
| avg_glucose_level | Blood glucose           |
| bmi               | Body Mass Index         |
| smoking_status    | Smoking history         |
| stroke            | Target variable (0 / 1) |

---

# 📊 Notebook Workflow (Cell-wise Explanation)

---

## 1️⃣ Importing Libraries

Libraries used:

* `pandas`, `numpy` → Data handling
* `matplotlib`, `plotly` → Visualization
* `sklearn` → ML models & preprocessing
* `imblearn` → SMOTE
* `optuna` → Hyperparameter tuning
* `pickle` → Model saving

---

## 2️⃣ Data Loading & Inspection

### Cells Performed:

* `df.shape`
* `df.info()`
* `df.isnull().sum()`
* `df.head()`

### Why:

To understand:

* Dataset size
* Data types
* Missing values
* Initial data structure

This step guides preprocessing decisions.

---

## 3️⃣ Outlier Detection

### Technique Used:

* Boxplots (Plotly)

### Why:

Outliers can distort model learning, especially in distance-based models.

Boxplots help visualize:

* Distribution spread
* Extreme values
* Skewness in features like BMI and age

---

## 4️⃣ Missing Value Treatment

### Steps Performed:

* Converted BMI to numeric
* Mean imputation for BMI
* Applied KNNImputer where necessary
* Removed remaining null rows

### Why:

Machine learning models cannot handle missing values directly.
Imputation ensures no data leakage and consistent model training.

---

## 5️⃣ Categorical Encoding

### Transformations:

* Gender → Binary encoding
* Ever_married → Binary encoding
* Residence_type → Binary encoding
* Dropped ID column
* Removed rare "Other" category

### Why:

ML models require numerical inputs.
Encoding ensures compatibility while preserving information.

ID column was removed because it carries no predictive value.

---

## 6️⃣ Target Variable Analysis

The dataset showed strong imbalance:

* ~95% No Stroke
* ~5% Stroke

### Why this matters:

Accuracy alone becomes misleading in imbalanced datasets.

Therefore, F1-score was considered important during evaluation.

---

## 7️⃣ Correlation & Multicollinearity Check

### Methods Used:

* Correlation matrix
* Heatmap
* Variance Inflation Factor (VIF)

### Why:

To detect:

* Highly correlated features
* Redundant predictors
* Multicollinearity issues

No severe multicollinearity was found.

---

## 8️⃣ Feature Selection

### Method Used:

Random Forest Feature Importance

### Selected Important Features:

* Age
* Gender
* BMI
* Residence_type

### Why:

Tree-based models provide feature importance scores.

Selecting top features:

* Reduces noise
* Improves generalization
* Prevents overfitting

---

## 9️⃣ Train-Test Split

```python
train_test_split(X, y, test_size=0.3, random_state=42)
```

### Why:

To evaluate model performance on unseen data and prevent overfitting.

---

## 🔟 Handling Class Imbalance – SMOTE

### Technique Used:

Synthetic Minority Oversampling Technique (SMOTE)

### Why:

The minority class (stroke cases) was underrepresented.

SMOTE:

* Generates synthetic minority samples
* Balances training data
* Improves recall and F1-score

SMOTE was applied **only on training data** to prevent data leakage.

---

## 1️⃣1️⃣ Model Training

Models trained:

* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gradient Boosting

Each model was implemented using a **Pipeline** including:

* StandardScaler
* Classifier

### Why StandardScaler:

Some algorithms (KNN, Gradient Boosting) perform better when features are scaled to similar ranges.

Pipeline ensures:

* No data leakage
* Reproducibility
* Clean workflow

---

## 1️⃣2️⃣ Hyperparameter Tuning – Optuna

Optuna was used to optimize model parameters.

### Why:

Manual tuning is inefficient and inconsistent.

Optuna:

* Automatically searches parameter space
* Uses intelligent optimization strategy
* Improves model performance systematically

Best performing model:
**Gradient Boosting (Optimized)**

---

## 1️⃣3️⃣ Model Evaluation

Metrics used:

* Accuracy
* F1 Score
* Classification Report

### Why F1-score:

Because dataset is imbalanced, F1-score balances Precision and Recall.

---

## 💾 Model Saving

```python
pickle.dump(best_pipeline, open("stroke_prediction_model.sav", "wb"))
```

### Why:

To:

* Reuse trained model
* Avoid retraining
* Enable deployment

---

# 🏁 Final Model

* Algorithm: Gradient Boosting
* Hyperparameters: Optimized using Optuna
* Imbalance Handling: SMOTE
* Evaluation Metric: F1 Score

The trained model is saved as:

```
stroke_prediction_model.sav
```

---

# 🔁 Reproducibility Notes

To run the project:

```bash
pip install pandas numpy scikit-learn imbalanced-learn optuna plotly
```

Then execute:

```
code.ipynb
```

---

# 👨‍💻 Author

**Darshan Nyati**
AI & Machine Learning Enthusiast



Tell me the style you prefer.
