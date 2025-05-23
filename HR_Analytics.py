import numpy as np
import pandas as pd

data=pd.read_csv('C:\\Users\\INSPIRON\\Desktop\\project\\dataset\\HR-Employee-Attrition.csv')
#print(data.head())

#print(data.dtypes)

#print(data.isnull().sum())

#attrition dependent on Age?
import plotly.express as px

age_att = data.groupby(['Age', 'Attrition']).size().reset_index(name='Counts')
fig = px.line(age_att,x='Age',y='Counts',color='Attrition',title='Agewise Counts of People in an Organization')
#fig.show()  

#income is the factor towards employee attrition?
rate_att=data.groupby(['MonthlyIncome','Attrition']).size().reset_index(name='Counts')
rate_att['MonthlyIncome']=round(rate_att['MonthlyIncome'],-3)
rate_att=rate_att.groupby(['MonthlyIncome','Attrition']).size().reset_index(name='Counts')
fig=px.line(rate_att,x='MonthlyIncome',y='Counts',color='Attrition',title='Monthly Income basis counts of People in an Organization')
#fig.show()

#Does the Department of work impact attrition?
dept_att=data.groupby(['Department','Attrition']).size().reset_index(name='Counts')
fig=px.bar(dept_att,x='Department',y='Counts',color='Attrition',title='Department wise Counts of People in an Organization')
#fig.show()

#environment satisfaction impact attrition?
satisfaction_att=data.groupby(['EnvironmentSatisfaction','Attrition']).size().reset_index(name='Counts')
fig = px.area(satisfaction_att,x='EnvironmentSatisfaction',y='Counts',color='Attrition',title='Environment Satisfaction level Counts of People in an Organization')
#fig.show()

#Job Satisfaction impact on Attrition
Job_Satisfaction_att=data.groupby(['JobSatisfaction','Attrition']).size().reset_index(name='Counts')
fig = px.area(Job_Satisfaction_att,x='JobSatisfaction',y='Counts',color='Attrition',title='Job Satisfaction level Counts of People in an Organization')
#fig.show()

#company stocks impact on employees attrition
stock_att = data.groupby(['StockOptionLevel', 'Attrition']).size().reset_index(name='Counts')
fig = px.bar(stock_att,x='StockOptionLevel',y='Counts',color='Attrition',title='Stock facilities level wise People in an Organization')
#fig.show()

#Work Life Balance impact on the overall attrition rates
Work_Life_Balance_att=data.groupby(['WorkLifeBalance','Attrition']).size().reset_index(name='Counts')
fig = px.bar(Work_Life_Balance_att,x='WorkLifeBalance',y='Counts',color='Attrition',title='Work Life Balance level Counts of People in an Organization')
#fig.show()

#work experience impact on attrition
work_experience_att=data.groupby(['NumCompaniesWorked','Attrition']).size().reset_index(name='Counts')
fig = px.area(work_experience_att,x='NumCompaniesWorked',y='Counts',color='Attrition',title='Work Experience level Counts of People in an Organization')
#fig.show()

#Work duration in current role impact on Attrition
current_role_att=data.groupby(['YearsInCurrentRole','Attrition']).size().reset_index(name='Counts')
fig = px.line(current_role_att,x='YearsInCurrentRole',y='Counts',color='Attrition',title='Counts of People working for years in an Organization')
#fig.show()

#Salary Hike percentage impact on Attrition
salary_hike_att=data.groupby(['PercentSalaryHike','Attrition']).size().reset_index(name='Counts')
fig = px.line(salary_hike_att,x='PercentSalaryHike',y='Counts',color='Attrition',title='Count of Hike Percentages people receive in an Organization')
#fig.show()  

#Are managers a reason of people resigning?
manager_att=data.groupby(['YearsWithCurrManager','Attrition']).size().reset_index(name='Counts')
fig = px.line(manager_att,x='YearsWithCurrManager',y='Counts',color='Attrition',title='Count of people spending years with a Manager in an Organization')
#fig.show()

#Job Level impact on Attrition
job_level_attr = data.groupby(['JobLevel', 'Attrition']).size().reset_index(name='Count')
fig = px.bar(job_level_attr, x='JobLevel', y='Count', color='Attrition',
             title='Attrition Count by Job Level')
#fig.show()

#Correlation Matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns from the dataset
numeric_data = data.select_dtypes(include=['float64', 'int64'])

corr_matrix = numeric_data.corr()

#plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.6)

#plt.title('Correlation Heatmap')
#plt.show()

# Set the threshold for high correlation
threshold = 0.8

# Create a mask to filter out the upper triangle and diagonal
mask = (corr_matrix.abs() > threshold) & (corr_matrix != 1)

# Extract the highly correlated pairs
highly_correlated_pairs = corr_matrix[mask]

#print("Highly Correlated Pairs:")
#print(highly_correlated_pairs.dropna(how='all', axis=0).dropna(how='all', axis=1))

# Plotting the relationship between JobLevel and MonthlyIncome
sns.scatterplot(x='JobLevel', y='MonthlyIncome', data=numeric_data)

# Adding title and labels
plt.title('Relationship between Job Level and Monthly Income')
plt.xlabel('Job Level')
plt.ylabel('Monthly Income')

# Show the plot
#plt.show()

fig = px.box(data,
             x='JobLevel',
             y='MonthlyIncome',
             color='Attrition',
             title='Monthly Income across Job Levels with Attrition',
             points="all")

#fig.show()

#Dropping irrelevant or unnecessary columns
# Make a copy of the data
df = data.copy()
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, errors='ignore', inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Logistic Regression
from sklearn.linear_model import LogisticRegression

# Encode 'Attrition' (Yes = 1, No = 0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col])

# Split features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']       #target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
#log_model = LogisticRegression(max_iter=1000)

# Train model
#log_model.fit(X_train, y_train)

#Evaluate the Model
#Predict on test set
#y_pred = log_model.predict(X_test)

#Evaluate
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))
#print(classification_report(y_test, y_pred))

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree model
#dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
#dt_model.fit(X_train, y_train)

#Evaluate the Model
# Predict the test set results
#y_pred = dt_model.predict(X_test)

# Accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))

# Confusion Matrix
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
#rf_model = RandomForestClassifier(random_state=42, n_estimators=100)        #n_estimators=100 means the forest will have 100 trees.

# Train the model
#rf_model.fit(X_train, y_train)

#Evaluate the Model
# Predict the test set results
#y_pred = rf_model.predict(X_test)

# Accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))

# Confusion Matrix
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Support Vector Machines (SVM)
from sklearn.svm import SVC

# Initialize the Support Vector Machine (SVM) model
#svm_model = SVC(kernel='linear', random_state=42)

# Train the model
#svm_model.fit(X_train, y_train)

#Evaluate the Model
# Predict the test set results
#y_pred = svm_model.predict(X_test)

# Accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))

# Confusion Matrix
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

#XGBoost (Extreme Gradient Boosting)
import xgboost as xgb

#Train the XGBoost model
#xgb_model = xgb.XGBClassifier()
#xgb_model.fit(X_train, y_train)

#Predict
#y_pred = xgb_model.predict(X_test)

#Evaluate
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))
#print(classification_report(y_test, y_pred))

#LightGBM
import lightgbm as lgb
'''
# Initialize the model
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=100,
    random_state=42
)
'''
from lightgbm import early_stopping, log_evaluation
'''
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=100,
    random_state=42
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    callbacks=[early_stopping(stopping_rounds=10), log_evaluation(0)]
)
'''

# Predict
#y_pred = lgb_model.predict(X_test)

#Evaluate
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))
#print(classification_report(y_test, y_pred))

#CatBoost
from catboost import CatBoostClassifier, Pool

# If you have categorical features, list their column indices
#cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]

# Initialize CatBoost
'''
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=0
)'''


# Fit model
#model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=10)

# Predict
#y_pred = model.predict(X_test)

#Evaluate
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))
#print(classification_report(y_test, y_pred))

#AdaBoost (Adaptive Boosting)
from sklearn.ensemble import AdaBoostClassifier

# Base learner
#base_tree = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoost
#ada_model = AdaBoostClassifier(estimator=base_tree, n_estimators=100, learning_rate=1.0, random_state=42)

# Train
#ada_model.fit(X_train, y_train)

# Predict
#y_pred = ada_model.predict(X_test)

#Evaluate
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy Score:", round(accuracy, 4))
#print(classification_report(y_test, y_pred))

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include='object'):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# this model has high accuracy





