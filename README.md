# **Maternal Health Risk Prediction (Matern_AI)**  

##  Introduction  
This project aims to develop and compare a vanilla Neural Network model to a model incorporating optimization techniques as well as Classical Algorithms to predict maternal health risk among pregnant women as either low, mid, or high risk.

More information about my proposal can be found here:  https://docs.google.com/document/d/1xnnj8wq3rHsqiU6VgBP0DfMDTO1EZ9WXXSz1G3YgdLc/edit?usp=sharing

##  About the Dataset  
The dataset used for this model is publicly available on Kaggle:  
🔗 [Maternal Health Risk Dataset](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data)  

### **Key Vairables:**  
- **Age** - This is age in years when a woman is pregnant. 
- **SystolicBP** - The upper value of Blood Pressure in mmHg.
- **DiastolicBP** - Lower value of Blood Pressure in mmHg.
- **BS (Blood Sugar Level)** - Blood glucose levels in terms of molar concentration, mmol/L.  
- **HeartRate** - A normal resting heart rate in beats per minute.
- **Risk Level** - Predicted Risk Intensity Level during pregnancy considering the previous attributes. Categorized as either 'High Risk', 'Low Risk', or 'Mid Risk'.

##  Data Cleaning & Preprocessing  
 **Handling Missing and Duplicate Data**  
   - No missing values were found in the dataset.  
   - 562 duplicate rows were identified and removed to enhance model accuracy.  

 **Outlier Detection & Correction**  
   - An unrealistic HeartRate value of 7 was detected and replaced with the mode (70 bpm).  

 **Encoding Categorical Variables**  
   - The Risk Level variable was encoded as follows:  
     - `2` → High Risk  
     - `1` → Mid Risk  
     - `0` → Low Risk  

 **Feature Correlation Analysis**  
   - A correlation heatmap revealed that Blood Sugar (BS) had the strongest positive correlation with Risk Level (0.55).  
   - Age and HeartRate showed weaker correlations but were retained as features.  

 **Handling Class Imbalance**  
   - **Class distribution after dropping duplicate rows:**  
     - 234 Low-Risk cases  
     - 106 Mid-Risk cases  
     - 112 High-Risk cases  
   - The imbalance was addressed through data scaling and appropriate model selection.  

 **Data Splitting & Scaling**  
   - The dataset was split into training and testing sets.  
   - Feature values were scaled for improved model performance.  

## Building Model using Classical ML Algorithms

### **Logistic Regression (Baseline Model)**
Best hyperparameters:
- Applied SMOTE to handle class imbalance before training.
- Defined the parameter grid (C and solver).
- Performed 5-fold cross-validation (cv = 5) to evaluate each combination.
- Selected the best model based on accuracy score.

Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.79     | 0.86   | 0.82     | 35      |
| 1     | 0.56     | 0.31   | 0.40     | 16      |
| 2     | 0.71     | 0.88   | 0.79     | 17      |
| **Accuracy** |  |  | **0.74** | 68 |
| **Macro Avg** | 0.69 | 0.68 | 0.67 | 68 |
| **Weighted Avg** | 0.72 | 0.74 | 0.71 | 68 |

### **XGBoost Classifier**
Hyperparameters Tuned:
- n_estimators (Number of Trees): [50, 100, 200] → Controls model complexity and training time.
- max_depth (Tree Depth): [3, 5, 7] → Determines tree complexity, balancing underfitting and overfitting.
- learning_rate (Step Size): [0.01, 0.1, 0.2] → Affects how quickly the model learns patterns.
- subsample (Row Sampling): [0.7, 0.8, 1.0] → Introduces randomness to reduce overfitting.



### Model Performance Comparison  

| Model                | Train Accuracy| Validation Accuracy | Test Accuracy| Precision | Recall | F1 Score |
|----------------------|---------------|---------------------|--------------|-----------|--------|----------|
| Logistic Regression  | 0.5853        | 0.6324              | 0.7353       |   | **0.85** | **0.86** |
| **XGBoost**         | **91.8%**     | **85.0%**            | **83.7%**    | **0.86**  | **0.84** | **0.85** |
| **Logistic Regression** | **78.4%**     | **76.2%**            | **75.1%**    | **0.74**  | **0.75** | **0.74** |
