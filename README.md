# **Maternal Health Risk Prediction (Matern_AI)**  

##  Introduction  
Maternal health complications are key challenges in many developing African countries, contributing to high mortality rates from preventable conditions. I aim to build a model that will analyze key indicators such as body temperature, heart rate, blood pressure, age, etc to assess risk levels among pregnant women, enabling them receive timely interventions in healthcare facilities.

More information about my proposal can be found here:  https://docs.google.com/document/d/1xnnj8wq3rHsqiU6VgBP0DfMDTO1EZ9WXXSz1G3YgdLc/edit?usp=sharing

In the project, I've developed and compared a vanilla Neural Network model to a model incorporating optimization techniques as well as Classical ML Algorithms to predict maternal health risk as either low, mid, or high risk.

##  About the Dataset  
The dataset used for this model is publicly available on Kaggle:  
🔗 [Maternal Health Risk Dataset](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data)  

### **Key Variables:**  
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
| Training Accuracy   | 0.6361 |
| Validation Accuracy | 0.6618 |
| Test Accuracy       | 0.6765 |
| Test Loss           | 0.8253 |

Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.64     | 0.97   | 0.77     | 35      |
| 1     | 0.50     | 0.12   | 0.20     | 16      |
| 2     | 0.91     | 0.59   | 0.71     | 17      |
|   Accuracy     |  |  |   0.65   |   68   |
|   Macro Avg    | 0.68 | 0.56 | 0.56 | 68 |
|   Weighted Avg   | 0.68 | 0.68 | 0.62 | 68 |


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
| Validation Accuracy  | 0.7059 |
| Test Accuracy        | 0.7353 |
| Test Loss            | 0.6958 |

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|  0    | 0.76     | 0.89   | 0.82     | 35      |
|  1    | 0.67     | 0.25   | 0.36     | 16      |
|  2  | 0.71     | 0.88   | 0.79     | 17      |
|   Accuracy   |  |  |   0.74   | 68 |
|   Macro Avg   | 0.71 | 0.67 | 0.66 | 68 |
|   Weighted Avg   | 0.72 | 0.74 | 0.70 | 68 |


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


### Summary of Findings
- The Vanilla Neural Network Model had two hidden layers (32 and 16 neurons) with ReLU activation but lacked optimization techniques. It achieved a test accuracy of 64.71%, showing signs of underfitting and struggling with class imbalance, especially for Class 1 (recall of 12%).
- The Adam Optimizer significantly improved performance, reaching 73.53% test accuracy. It used L2 regularization (0.001), dropout (0.1), dynamic learning rate, and early stopping. The model generalized better, but Class 1 recall remained relatively low (31%).
- The SGD Optimizer performed best, achieving 76.47% test accuracy. It benefited from L2 regularization (0.005), a very small learning rate (0.00001), early stopping (patience 5), and a larger batch size (64). It had the highest recall for Class 1 (38%) and balanced performance across all classes.
- The RMSProp Optimizer had the lowest test accuracy among optimized models (70.59%). Despite L2 regularization and a fixed learning rate (0.0005), it struggled with low recall for Class 1 (31%), showing less stability than SGD and Adam.
- SGD was the best-performing optimizer, providing the highest accuracy and better recall for the minority class. However, class imbalance remains an issue.


## Model Recommendation
Among all models, the **SGD-optimized Neural Network** performed the best with 0.7647 test accuracy and the highest recall for Class 1 (0.38). 

**XGBoost** was the best classical ML model, but it was slightly outperformed by the optimized neural network. 

The choice of which model to use depends on whether a simpler model (XGBoost) or a more complex but better-performing model (SGD-optimized Neural Network) is preferred.

## Running my Notebook and Accessing Saved Models
- Open the notebook and run all the cells from the section titled "Dataset Loading" up to the "Data Preprocessing" section. This includes splitting the data and scaling it appropriately.
- Once the preprocessing is complete, proceed to the section labeled "TESTING MY MODELS" and run the corresponding cell. This cell contains pre-written code to load the saved models and facilitate prediction. For example, for our best model using SGD Optimizer:
  ```python
     from tensorflow.keras.models import load_model
     import numpy as np
     
     model = load_model("maternal_health_sgd.keras")
     nn_probs = sgd_model.predict(X_test)
     if nn_probs.shape[1] > 1:
        y_test_pred_nn_labels = nn_probs.argmax(axis=1)
      else:
          y_test_pred_nn_labels = (nn_probs > 0.5).astype(int)

     print("Predictions (First 10):", y_test_pred_nn_labels[:10].flatten())
     ```
  
- Ensure that both the model files and the dataset are uploaded to the appropriate directories for smooth execution.

## Conclusion
This project successfully demonstrates how machine learning can be used for maternal health risk prediction. The optimized **SGD Neural Network** showed the best performance, making it the recommended model for deployment.

## Demo Video
https://youtu.be/QHtpgDP0OaM
