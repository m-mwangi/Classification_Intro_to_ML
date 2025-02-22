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
