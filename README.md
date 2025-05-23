# 💼HR ANALYTICS FOR EMPLOYEE ATTRITION PREDICTION
### 📘Introduction:
HR Analytics leverages data-driven insights to improve workforce management. One critical application is predicting employee attrition—understanding why employees leave and identifying those at risk of leaving. This helps organizations retain talent, reduce turnover costs, and enhance productivity.
### ❓Problem Statement:
The goal is to develop a predictive model using employee data to identify individuals likely to leave the organization. By analyzing factors like job role, satisfaction, workload, and compensation, HR teams can take proactive steps to improve retention.
### 📊Data Acquisition
The data for the project is sourced from Kaggle and is made available to everyone. This information includes **1,470 records**, each containing **35 details** regarding employees. <br> 
> ✅ The dataset is already included in the repository.[Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) .
### 🔄 Preprocessing
The dataset is preprocessed to handle missing values, encode categorical features, and scale numerical values.
### 🤖Model Development
We implemented and compared a range of machine learning models, including:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **LightGBM**
- **CatBoost**
- **AdaBoost**
- **Gradient Boosting**

Each model was trained on the same dataset and tested using consistent evaluation metrics.
## 📈 Model Evaluation
To assess performance, we used the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

<br>
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 83.21%   | 78.00%    | 71.00% | 74.30%   |
| Decision Tree       | 84.08%   | 79.10%    | 73.50% | 76.20%   |
| Random Forest       | 87.20%   | 83.00%    | 75.60% | 79.10%   |
| SVM                 | 86.15%   | 82.00%    | 74.20% | 77.90%   |
| XGBoost             | 88.40%   | 85.10%    | 77.00% | 80.80%   |
| LightGBM            | 87.65%   | 84.00%    | 76.10% | 79.80%   |
| CatBoost            | 88.70%   | 85.60%    | 78.00% | 81.60%   |
| AdaBoost            | 85.95%   | 81.50%    | 74.30% | 77.70%   |
| **Gradient Boosting** | **89.12%** | **86.00%** | **79.30%** | **82.50%** |

### 🏆Best Model Selection
After the evaluation, it was determined that the **gradient boosting algorithm** performed the best. > 🎯 **Accuracy: 89.12%** <br>
So we trained the [model](https://github.com/shruti041/HR_ANALYTICS_FOR_EMPLOYEE_ATTRITION_PREDICTION/blob/main/Gradient_Boosting.py) and save the model and label encoders. <br>
### 🧪Simulation Environment
- **IDE**: Visual Studio Code (VSCode) for development and debugging.
- **Frontend**: Built using **Streamlit** for its simplicity and fast deployment.[Streamlit_code](https://github.com/shruti041/HR_ANALYTICS_FOR_EMPLOYEE_ATTRITION_PREDICTION/blob/main/Final_app.py).
## ❔Why Gradient Boosting tends to perform well?
Gradient Boosting often performs better in employee attrition prediction because it is capable of modeling non-linear, complex relationships, handles imbalanced data relatively well, and focuses on learning from previous mistakes, which helps capture the nuances in attrition behavior.
