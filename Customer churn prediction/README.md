# Customer Churn Prediction

Predicting customer behavior using machine learning to identify at-risk telecom customers and enable proactive retention strategies.

---

## 📋 Overview

Customer churn—when customers discontinue service—is a critical metric in the telecom industry. Acquiring new customers costs 5-25x more than retaining existing ones, making churn prediction invaluable for business strategy.

This project builds and compares multiple classification models to predict which customers are likely to churn, enabling data-driven retention decisions.

---

## 📊 Dataset

**Source:** IBM Telco Customer Churn Dataset (Kaggle)

- **Samples:** 7,043 customers
- **Features:** 21 customer attributes
- **Target:** Binary churn classification (Yes/No)

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| Tenure | Numeric | Months as a customer |
| Monthly Charges | Numeric | Monthly subscription cost |
| Total Charges | Numeric | Total lifetime charges |
| Contract Type | Categorical | Month-to-month, 1-year, or 2-year |
| Internet Service | Categorical | Fiber optic, DSL, or no service |
| Senior Citizen | Binary | Whether customer is 65+ |

---

## 🔧 Technologies Used

- **Python 3.x**
- **Scikit-learn** — ML models and preprocessing
- **XGBoost** — gradient boosting classifier
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib & Seaborn** — data visualization

---

## 🛠️ Data Pipeline

### 1. Data Cleaning
- Handled missing values in `TotalCharges` (removed ~11 rows)
- Removed zero-variance features
- Type conversion for numeric columns

### 2. Preprocessing
- **Categorical encoding:** One-hot encoding for nominal variables
- **Feature scaling:** StandardScaler for numerical features
- **Class imbalance:** SMOTE oversampling applied in training pipeline

### 3. Train-Test Split
- 80% training / 20% test stratified by target class
- Stratification preserves churn distribution across folds

---

## 🎯 Key Findings

- **Month-to-month contracts** have higher churn rate than long-term contracts
- **Longer tenure** strongly correlates with retention (new customers churn 25x more)
- **Higher monthly charges** weakly associated with increased churn risk

---

## 📈 Model Performance

### Cross-Validation Results (5-Fold Stratified)

| Metric | Random Forest | XGBoost |
|---|---|---|
| **F1 Score** | 0.4502 | 0.4856 |
| **ROC-AUC** | 0.8162 | 0.7997 |
| **Precision** | 0.7768 | 0.5591 |
| **Recall** | 0.3316 | 0.4368 |

### Model Comparison

**Random Forest:**
- Strengths: High precision (0.78) — few false alarms
- Weakness: Low recall (0.33) — misses many actual churners
- Use case: Conservative approach; minimize false retention spend

**XGBoost:**
- Strengths: Balanced metrics; catches more churners (0.44 recall)
- Weakness: Lower precision (0.56) — more false alarms
- Use case: Aggressive retention; maximize customer capture

### Test Set Results (After Threshold Tuning)

**XGBoost (Threshold = 0.15):**
- Accuracy: 76%
- Precision (Churn class): 36%
- Recall (Churn class): 66%
- F1 Score: 0.46

**Key Insight:** XGBoost identifies 66% of customers who will actually churn, making it more suitable for proactive outreach campaigns.

### Threshold Optimization

Classification threshold was tuned across 28 values to maximize F1 on the minority (churn) class:
- **Random Forest:** Best threshold = 0.275 (F1 = 0.5528)
- **XGBoost:** Best threshold = 0.150 (F1 = 0.4627)

Lower threshold for XGBoost reflects its more aggressive identification of churn risk.

---

## 💾 Model Deployment

The trained model is serialized using Pickle for production inference:

```python
import pickle

# Load trained model
with open('xgboost_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new customer
new_customer = [[...features...]]
churn_probability = model.predict_proba(new_customer)[0, 1]
prediction = "Churn Risk" if churn_probability > 0.15 else "Safe"
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
 [  git clone <(https://github.com/TamahiAlmutassim/Python-Projects/tree/main/Customer%20churn%20prediction)>
   cd customer-churn-prediction]
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook notebooks/churn_analysis.ipynb
   ```

4. **Load and use the model**
   ```python
   import pickle
   with open('models/xgboost_churn_model.pkl', 'rb') as f:
       model = pickle.load(f)
   predictions = model.predict_proba(X_new)
   ```

---

## Findings

✅ SMOTE effectively balances imbalanced datasets in ML pipelines  
✅ Threshold tuning can significantly improve minority class performance  
✅ Cross-validation with stratification is critical for imbalanced data  
✅ Model selection depends on business context (precision vs. recall trade-off)  

---

## Author

**Tamahi Almutassim**  
[LinkedIn](https://www.linkedin.com/in/tamahi-almutassim-9a2279158/) · [GitHub](https://github.com/TamahiAlmutassim)

---

## 📄 License

Dataset source: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn) — Open Database License
