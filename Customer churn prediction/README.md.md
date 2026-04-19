Customer churn prediction

predicting customers behaviour using machine learning 

-----------------------------------------------------

Overview

\## 



Customer churn refers to customers who stop using a company's services. In the telecom industry, retaining customers is more cost-effective than acquiring new ones.



This project builds a machine learning model to predict whether a customer is likely to churn based on data. The goal is to help telecom companies identify at-risk customers early and take preventive actions.

-----------------------------------------------------

&nbsp;📊 Dataset



\- Source: Kaggle Telco Customer Churn Dataset

\- Rows: ~7,043 customers

\- Features: 21 customer attributes

important Features:

\- Gender

\- SeniorCitizen

\- Tenure

\- Contract type

\- MonthlyCharges

\- TotalCharges

\- InternetService

\- Churn (Target variable)

-----------------------------------------------------

Data Cleaning:

\- Handled missing values in TotalCharges

\- Encoded categorical variables using Label Encoding
-----------------------------------------------------


Workflow

The project follows a standard machine learning pipeline:

1\. Data Collection

2\. Data Cleaning \& Preprocessing

3\. Exploratory Data Analysis (EDA)

4\. Feature Engineering

5\. Train-Test Split

6\. Model Training

7\. Model Evaluation

8\. Model Optimization

-----------------------------------------------------

Founding:

\- Customers with month-to-month contracts are more likely to churn

\- Higher monthly charges increase churn probability

\- Longer tenure significantly reduces churn risk

-----------------------------------------------------

**Models used** 

\- Logistic Regression

\- Decision Tree Classifier

\- Random Forest Classifier

\- XGBoost Classifier

---

Model comparison

| Model               | F1 Score |

|---------------------|----------|

| Decision tree       | 0.80     |

| Random Forest       | 0.84     |

| XGBoost             | 0.83     |

&nbsp;





