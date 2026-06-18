# CodeAlpha_Credit_Scoring_Model

# End-to-End Credit Scoring Classification Pipeline

## Project Overview
This repository contains a professional-grade, end-to-end Machine Learning pipeline designed to evaluate an individual's creditworthiness. Utilizing historical financial indicators, the system handles missing data imputation, continuous variable scaling, categorical encoding, advanced domain-specific feature engineering, class imbalance mitigation, and exhaustive hyperparameter tuning. 

The primary business objective is to accurately classify applicants into **Good Credit (Low Risk)** or **Bad Credit (High Risk)** tiers, providing automated decision support to minimize financial defaults.

---

## Dataset Specifications
The pipeline utilizes the classic **German Credit Risk Dataset** (sourced via Kaggle from the UCI Machine Learning Repository). 
* **Total Instances:** 1,000 applicants
* **Target Feature:** `Risk` (`good` or `bad`)
* **Input Attributes:**
  * **Numerical:** `Age`, `Credit amount`, `Duration` (in months)
  * **Categorical:** `Sex`, `Job` (skill levels 0–3), `Housing` (own, rent, free), `Saving accounts`, `Checking account`, and loan `Purpose` (car, business, education, etc.)

---

## Data Preprocessing & Pipeline Architecture

A rigorous data processing workflow was built using `scikit-learn`'s `ColumnTransformer` to enforce a clean separation between training patterns and validation data:

1. **Missing Value Imputation:** Missing entries in `Saving accounts` and `Checking account` were explicitly filled with an `'unknown'` flag, treating the absence of an account as a distinct risk characteristic rather than dropping rows.
2. **Feature Mapping:** The categorical text target variable was mapped to a clean binary format: `good` $\rightarrow 0$ and `bad` $\rightarrow 1$.
3. **Continuous Feature Standardization:** Numerical columns (`Age`, `Credit amount`, `Duration`) were normalized using `StandardScaler` to have a zero mean and unit variance ($\mu = 0, \sigma = 1$).
4. **Categorical Encoding:** Categorical attributes were transformed via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, expanding the feature space from the initial 10 attributes to **29 independent variables** to cleanly capture variations across categorical groups.
5. **Stratified Data Splitting:** The data was partitioned into an **80% Training Set (800 samples)** and a **20% Test Set (200 samples)**. Stratification was strictly enforced (`stratify=y`) to maintain a uniform balance of good vs. bad credit risk profiles across both divisions.

---

## Evolutionary Model Training & Optimization Iterations

### Phase 1: Baseline Architecture Comparison
Three distinct core classification models were initially evaluated using default parameters to establish a performance baseline:
* **Logistic Regression** (Linear baseline)
* **Decision Tree Classifier** (Non-linear baseline, constrained to `max_depth=5`)
* **Random Forest Classifier** (Ensemble baseline consisting of 100 estimators)

#### Baseline Results:
* **Logistic Regression:** Achieved an overall accuracy of **73%** and an ROC-AUC of **0.7613**. However, it suffered from severe class bias due to unmitigated dataset imbalance, yielding a poor recall of **38%** for high-risk applicants.
* **Decision Tree:** Registered an accuracy of **68%** and an ROC-AUC of **0.7399**, catching more defaults (**53% recall**) but introducing substantial variance.
* **Random Forest:** Formed the strongest baseline with an accuracy of **74%** and an ROC-AUC of **0.7784**, though it remained constrained by a lower recall of **40%** on high-risk profiles.

---

### Phase 2: Advanced Engineering & Optimization
To address the majority class bias and maximize credit default catching capabilities, three major software engineering enhancements were implemented:

1. **Domain Feature Engineering:** Extracted mathematical proxy indicators from financial histories to help tree algorithms recognize risk thresholds:
   * `Monthly_Burden` = $\frac{\text{Credit amount}}{\text{Duration}}$ (Simulates immediate repayment pressure)
   * `Credit_to_Age_Ratio` = $\frac{\text{Credit amount}}{\text{Age}}$ (Scales risk relative to life/employment stage)
2. **Synthetic Minority Over-sampling Technique (SMOTE):** Artificially balanced the structural distribution of training features (`X_train_balanced`, `y_train_balanced`) to prevent the classifiers from ignoring minority high-risk signals.
3. **Exhaustive Structural Fine-Tuning (`GridSearchCV`):** Swapped random parameters for an automated, 5-fold cross-validated grid sweep explicitly optimizing for a balanced **`f1_macro`** metric. The search spaces evaluated forest sizing, splitting criteria (`gini` vs `entropy`), depth bounds, and leaf node minimum splits.

---

## Final Model Evaluation Metrics

Following deep hyperparameter optimization, the ideal architecture configuration was selected:
`{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 18, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`

### Final Classification Report (Test Set):

| Credit Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Good Credit (Class 0)** | 0.81 | 0.82 | 0.82 | 140 |
| **Bad Credit (Class 1)** | 0.57 | 0.55 | 0.56 | 60 |
| **Overall Accuracy** | | | **0.74** | **200** |
| **Macro Average** | 0.69 | 0.69 | 0.69 | 200 |
| **Weighted Average** | 0.74 | 0.74 | 0.74 | 200 |

* **Peak ROC-AUC Score:** **0.7814**

### Optimized Confusion Matrix Breakdown:
```text
[[115  25]   -->  [ [ True Negatives (Correct Approved),  False Positives (Missed Bad) ]
 [ 27  33]]   -->    [ False Negatives (Missed Good),     True Positives (Correct Caught) ] ]
