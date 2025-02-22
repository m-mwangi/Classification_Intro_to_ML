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

### **Logistic Regression**
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
|   Accuracy   |  |  |    0.74 | 68 |
|   Macro Avg   | 0.69 | 0.68 | 0.67 | 68 |
|   Weighted Avg   | 0.72 | 0.74 | 0.71 | 68 |

### **XGBoost Classifier**
Hyperparameters Tuned:
- n_estimators (Number of Trees): [50, 100, 200] → Controls model complexity and training time.
- max_depth (Tree Depth): [3, 5, 7] → Determines tree complexity, balancing underfitting and overfitting.
- learning_rate (Step Size): [0.01, 0.1, 0.2] → Affects how quickly the model learns patterns.
- subsample (Row Sampling): [0.7, 0.8, 1.0] → Introduces randomness to reduce overfitting.

Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|  0    | 0.77     | 0.94   | 0.85     | 35      |
|  1    | 0.50     | 0.25   | 0.33     | 16      |
|  2    | 0.76     | 0.76   | 0.76     | 17      |
|   Accuracy   |  |  |   0.74   | 68 |
|   Macro Avg   | 0.68 | 0.65 | 0.65 | 68 |
|   Weighted Avg   | 0.70 | 0.74 | 0.71 | 68 |


### Performance Comparison Between the 2 Models

| Model                | Train Accuracy | Validation Accuracy | Test Accuracy|
|----------------------|--------------- |---------------------|--------------|
| Logistic Regression  | 0.5853         | 0.6324              | 0.7353       |
| XGBoost              | 0.7279         | 0.7647              | 0.7353       |


### Key Observations:
- Both models achieve similar test accuracy (0.7353), but XGBoost has higher train and validation accuracy, indicating that it generalizes better.
- Logistic Regression shows a larger gap between training (0.5853)and test accuracy (0.7353), which might indicate that it struggles to learn complex patterns but generalizes well.
- Class-Wise Performance:
  - Class 0: Both models perform well.
  - Class 1: Both struggle, but Logistic Regression has a slightly better recall of 0.31 as compared to XGBoost with 0.25.
  - Class 2: Both models perform well.
- Due to the imbalanced nature of the dataset, there is evidence of bias, as indicated by the two models finding it difficult to predict class 1.

## Building Neural Network Models

### **Vanilla Neural Network Model**
This was a basic model with no optimization techniques. It consists of two hidden layers with ReLu activation(32 and 16 neurons, respectively). The output layer has a softmax activation function to handle multi-class classification.

Model Performance:

| Metric              | Value  |
|---------------------|--------|
| Training Accuracy   | 0.6551 |
| Validation Accuracy | 0.6471 |
| Test Accuracy       | 0.6471 |
| Test Loss           | 0.7883 |

Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.64     | 0.91   | 0.75     | 35      |
| 1     | 0.50     | 0.12   | 0.20     | 16      |
| 2     | 0.71     | 0.59   | 0.65     | 17      |
|   Accuracy     |  |  |   0.65   |   68   |
|   Macro Avg    | 0.62 | 0.54 | 0.53 | 68 |
|   Weighted Avg   | 0.63 | 0.65 | 0.60 | 68 |


Key Observations:
- The test accuracy is close to the validation accuracy, indicating that the model generalizes similarly on unseen data. However, the overall accuracy is not particularly high, suggesting room for improvement.
- Class imbalance greatly affects the performance, especially in class 1.
- There is potential underfitting and the model failing to capture complex patterns due to the low overall accuracy and poor recall for Class 1.
- With optimization, performance may be improved.

### **Optimized Neural Network Models**
To enhance the vanilla model’s performance, various optimization techniques were applied. I used the three optimizers: Adam, SGD, and RMSProp. I also used L2 regularization, Early Stopping, Learning Rate, and Patience Value. These played a vital role in improving my model's performance. 

Below are the findings for using each optimizer:

#### a) Implementing Adam Optimizer
Model Performance:

| Metric               | Value  |
|----------------------|--------|
| Training Accuracy    | 0.7152 |
| Validation Accuracy  | 0.7353 |
| Test Accuracy        | 0.7353 |
| Test Loss            | 0.6641 |

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|  0    | 0.79     | 0.89   | 0.84     | 35      |
|  1    | 0.50     | 0.31   | 0.38     | 16      |
|  2  | 0.74     | 0.82   | 0.78     | 17      |
|   Accuracy   |  |  |   0.74   | 68 |
|   Macro Avg   | 0.68 | 0.67 | 0.67 | 68 |
|   Weighted Avg   | 0.71 | 0.74 | 0.72 | 68 |


#### b) Implementing SGD Optimizer
Model Performance:

| Metric             | Value  |
|--------------------|------  |
| Training Accuracy  | 0.7184 |
| Validation Accuracy| 0.7059 |
| Test Accuracy      | 0.7647 |
| Test Loss          | 0.6865 |

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.79     | 0.89   | 0.84     | 35      |
| 1     | 0.60     | 0.38   | 0.46     | 16      |
| 2     | 0.79     | 0.88   | 0.83     | 17      |
|   Accuracy   |  |  |   0.76  | 68 |
|   Macro Avg  | 0.73 | 0.71 | 0.71 | 68 |
|   Weighted Avg | 0.75 | 0.76 | 0.75 | 68 |


#### c) Implementing RMSProp Optimizer
Model Performance:

| Metric               | Value  |
|----------------------|--------|
| Training Accuracy    | 0.7247 |
| Validation Accuracy  | 0.7206 |
| Test Accuracy        | 0.7059 |
| Test Loss            | 0.6835 |

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.75     | 0.86   | 0.80     | 35      |
| 1     | 0.45     | 0.31   | 0.37     | 16      |
| 2     | 0.76     | 0.76   | 0.76     | 17      |
| Accuracy  |   |   | 0.71 | 68 |
| Macro Avg | 0.66 | 0.64 | 0.65 | 68 |
| Weighted Avg | 0.68 | 0.71 | 0.69 | 68 |

### Summary of Optimizer and Hyperparameter Tuning

| Training Instance | Optimizer Used | Regularizer | Epochs | Early Stopping | Number of Layers | Learning Rate | Dropout Rate | Batch Size |
|------------------|---------------|------------|--------|---------------|---------------|--------------|-------------|-----------|
| 1               | Adam (default) | None       | 10     | No            | 3             | 0.001 (default) | None        | 32        |
| 2               | Adam          | L2 (0.001) | 150    | Yes (8 patience) | 3             | 0.001 (Reduces dynamically) | 0.1         | 32        |
| 3               | SGD           | L2 (0.005) | 100    | Yes (5 patience) | 3             | 0.00001 (Reduces dynamically) | 0.1         | 64        |
| 4               | RMSProp       | L2 (0.0001) | 150    | Yes (5 patience) | 3             | 0.0005       | 0.1         | 32        |
