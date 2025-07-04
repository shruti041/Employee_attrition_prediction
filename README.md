# 💼HR ANALYTICS FOR EMPLOYEE ATTRITION PREDICTION
### 📘Introduction:
HR Analytics leverages data-driven insights to improve workforce management. One critical application is predicting employee attrition—understanding why employees leave and identifying those at risk of leaving. This helps organizations retain talent, reduce turnover costs, and enhance productivity.
### ❓Problem Statement:
The goal is to develop a predictive model using employee data to identify individuals likely to leave the organization. By analyzing factors like job role, satisfaction, workload, and compensation, HR teams can take proactive steps to improve retention.
### 📊Data Acquisition
The data for the project is sourced from Kaggle and is made available to everyone. This information includes **1,470 records**, each containing **35 details** regarding employees. <br> 
> ✅ The dataset is already included in the repository.[Dataset](https://github.com/shruti041/HR_ANALYTICS_FOR_EMPLOYEE_ATTRITION_PREDICTION/blob/main/HR-Employee-Attrition.csv) .
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
### 🏆Best Model Selection
After the evaluation, it was determined that the **gradient boosting algorithm** performed the best. 
> 🎯 **Accuracy: 89.12%** <br>
So we trained the [model](https://github.com/shruti041/HR_ANALYTICS_FOR_EMPLOYEE_ATTRITION_PREDICTION/blob/main/Gradient_Boosting.py) and save the model and label encoders. <br>
### 🧪Simulation Environment
- **IDE**: Visual Studio Code (VSCode) for development and debugging.
- **Frontend**: Built using **Streamlit** for its simplicity and fast deployment.[Streamlit_code](https://github.com/shruti041/HR_ANALYTICS_FOR_EMPLOYEE_ATTRITION_PREDICTION/blob/main/Final_app.py).
### 🛠️ Retention Recommendations (App Feature)
One of the standout features of this application is the personalized recommendation system that activates when an employee is predicted to be at high risk of attrition.
<br>Based on key input indicators, the app provides specific, actionable suggestions that HR teams can use to proactively retain employees. These recommendations are dynamically generated and tailored to the individual's profile.
## ❔Why Gradient Boosting tends to perform well?
Gradient Boosting often performs better in employee attrition prediction because it is capable of modeling non-linear, complex relationships, handles imbalanced data relatively well, and focuses on learning from previous mistakes, which helps capture the nuances in attrition behavior.
