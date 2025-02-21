** Maternal Health Risk Prediction Model**

Introduction
The project aims to develop and compare a vanilla Neural network model to a model that uses optimization techniques to predict maternal risk as either low, mid, or high risk.

About the dataset
The dataset used for this model is publicly available on Kaggle: https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data

The variables include:

Age: Age in years when the woman is pregnant.

SystolicBP: Upper value of blood pressure (in mmHg), a significant attribute during pregnancy.

DiastolicBP: Lower value of blood pressure (in mmHg).

BS: Blood glucose levels measured in mmol/L.

HeartRate: Normal resting heart rate in beats per minute.

Risk Level: Predicted risk intensity level during pregnancy (Low, Mid, High).

Data Cleaning and Preprocessing.
The dataset was first loaded, and no missing values were found. However, 562 duplicate rows were identified and removed to ensure accurate model performance. Upon inspecting the data, an unrealistic HeartRate value of 7 was detected and replaced with the mode, 70.

The RiskLevel variable was encoded as follows: 2 for "high risk" 1 for "mid risk" 0 for "low risk"

A correlation heatmap revealed that Blood Sugar (BS) had the strongest positive correlation with RiskLevel (0.55), while Age and HeartRate had weaker correlations. Despite this, all the features were retained. There is a class imbalance (406 low-risk, 336 mid-risk, 272 high-risk). The data was split into training and testing sets and then scaled, making it ready for model training.
