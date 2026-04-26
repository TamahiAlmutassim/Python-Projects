# Employee Attrition Intelligence
### EDA & Predictive Modeling on IBM HR Data

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## Overview

Employee attrition — the rate at which staff voluntarily leave an organization — is one of the most costly and disruptive challenges in HR management. This project combines **exploratory data analysis** and **machine learning** to identify the key drivers of attrition and build a model capable of flagging at-risk employees before they resign.

The analysis is built on the [IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), a widely used benchmark containing demographic, role, compensation, and satisfaction data for 1,470 employees.

---

## Business Question

> *Which employees are most likely to leave, and what organizational factors drive that decision?*

Answering this allows HR teams to prioritize retention interventions — compensation reviews, role changes, workload adjustments — on the employees most at risk.

---

## Project Structure

```
hr-attrition-predictor/
│
├── HR_attrition_improved.ipynb   # Main analysis notebook
├── README.md                     # This file
└── data/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Source dataset (IBM / Kaggle)
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution of attrition across **Department**, **Job Role**, **Education Field**, **Gender**, and **Marital Status**
- Continuous variable analysis: **Age**, **Monthly Income**, **Distance from Home**, **Total Working Years**, **Percent Salary Hike**
- Discrete variable analysis: **Job Level**, **Environment Satisfaction**, **Job Involvement**, **Stock Option Level**, **Performance Rating**
- Identified class imbalance: ~84% "Stay" vs ~16% "Leave"

### 2. Preprocessing Pipeline 
All transformations are encapsulated in a single `imblearn.Pipeline` + `ColumnTransformer`, ensuring that no information from the test set leaks into training

| Column Type | Strategy |
|---|---|
| Numeric | Median imputation → Standard scaling |
| Ordinal (`BusinessTravel`) | Mode imputation → Ordinal encoding |
| Nominal (all others) | Mode imputation → One-Hot encoding |
| Class imbalance | SMOTE applied inside the pipeline on training data only |

### 3. Model Training & Selection
Two classifiers were trained and compared:
- **Random Forest** (300 estimators, `class_weight="balanced"`)
- **XGBoost** (300 estimators, `scale_pos_weight` for imbalance)

### 4. Evaluation
- **Stratified 5-Fold Cross-Validation** for reliable, variance-aware performance estimates
- **Threshold tuning** across 28 values (0.05 → 0.75) to maximize F1 on the minority class
- **ROC-AUC** and **Precision-Recall** curves with baselines
- Confusion matrix and full classification report on hold-out test set

---

## Key Findings

### EDA Insights
| Factor | Finding |
|--------|---------|
| **Overtime**       | Employees working overtime are disproportionately likely to leave |
| **Monthly Income** | Median income for leavers (~$2,500–3,000) vs. stayers (~$5,000+) |
| **Age**            | The 25–32 age group shows the highest attrition rate |
| **Job Role**       | Sales Representatives have the highest attrition percentage |
| **Marital Status** | Single employees leave at significantly higher rates |
| **Job Level**      | Lower job levels correlate strongly with higher attrition |

### Model Performance (5-Fold CV)

| Metric | Random Forest | XGBoost|
|---        |---         |---   
| F1 Score  |   0.4502   | 0.4856 |
| ROC-AUC   |   0.8162   | 0.7997 |
| Precision |   0.7768   | 0.5591 |
| Recall    |   0.3316   | 0.4368 |


### Top Predictive Features
1 Overtime
2 Stock option
3 Job level 


---

## Business Recommendation

Based on both the EDA and feature importance results, retention efforts should be prioritized on employees who are:
- **Working overtime regularly**
- **with low job level
- **In the Sales Representative or Lab Technician roles**
- **Who do not have stock option
- **Earning below the median monthly income**

A targeted intervention program for these segments — including compensation reviews, overtime caps, and career path discussions — could meaningfully reduce voluntary attrition.

---
##challenges 
-Handling the class imbalance (16% attrition) was tricky; SMOTE helped, but I had to be careful with the classification threshold to avoid too many false positives.

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, pipelines, evaluation |
| `imbalanced-learn` | SMOTE, imbalanced pipeline |
| `xgboost` | Gradient boosting classifier |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/hr-attrition-predictor.git
cd hr-attrition-predictor

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

# 3. Add the dataset to data/
#    Download from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# 4. Open the notebook
jupyter notebook HR_attrition_improved.ipynb
```

Or open directly in **Google Colab** — update the file path in cell 2 to point to your uploaded CSV.

---

## Author

**[Tamah Almutassim Alhaj]**  
[LinkedIn](https://www.linkedin.com/in/tamahi-almutassim-9a2279158/) · [GitHub](https://github.com/TamahiAlmutassim)

---

*Dataset source: IBM HR Analytics — made available via Kaggle under the Open Database License.*
