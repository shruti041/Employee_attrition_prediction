import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("gb_attrition_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.markdown("""
    <h1 style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #C0392B;
        text-shadow: 2px 2px 4px #aaa;
        font-weight: bold;
        letter-spacing: 2px;
        text-align: center;
        ">
        Employee Attrition Prediction
    </h1>
    <p style="
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        color: #333333;
        margin-top: -20px;
        margin-bottom: 30px;
        text-align: center;
        ">
        Welcome to the <strong>Employee Attrition Prediction</strong> tool.<br>
        Use this app to assess the likelihood of an employee leaving based on various factors.<br>
        Fill in the details and click <strong>Predict</strong> to get insights.<br>
        This helps HR proactively engage employees and reduce turnover.
    </p>
    """, unsafe_allow_html=True)

# Input UI
def user_input_features():
    Age = st.slider("Age", 18, 60, 30)
    BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    DailyRate = st.slider("Daily Rate", 102, 1499, 800)
    Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    DistanceFromHome = st.slider("Distance from Home", 1, 29, 10)
    education_options = {
    "1 - Below College": 1,
    "2 - College": 2,
    "3 - Bachelor": 3,
    "4 - Master": 4,
    "5 - Doctor": 5
    }
    Education_label = st.selectbox("Education Level", list(education_options.keys()))
    EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    EnvironmentSatisfaction = st.slider("Environment Satisfaction (1=Low to 4=Very High)", 1, 4, 3)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    HourlyRate = st.slider("Hourly Rate", 30, 100, 65)
    JobInvolvement = st.slider("Job Involvement (1=Low to 4=Very High)", 1, 4, 3)
    job_level_options = {
    "1 - Entry Level": 1,
    "2 - Junior": 2,
    "3 - Mid Level": 3,
    "4 - Senior": 4,
    "5 - Executive": 5
    }
    JobLevel_label = st.selectbox("Job Level", list(job_level_options.keys()))
    JobRole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"])
    JobSatisfaction = st.selectbox("Job Satisfaction (1=Low to 4=Very High)", [1, 2, 3, 4])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
    MonthlyRate = st.slider("Monthly Rate", 2094, 26999, 14000)
    NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 9, 2)
    OverTime = st.selectbox("Over Time", ["Yes", "No"])
    PercentSalaryHike = st.slider("Percent Salary Hike", 11, 25, 15)
    PerformanceRating = st.selectbox("Performance Rating", [3, 4], index=0)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction (1=Low to 4=Very High)", 1, 4, 3)
    stock_option_levels = {
    "0 - No stock options": 0,
    "1 - Basic/Entry-level stock option package": 1,
    "2 - Moderate-level stock options": 2,
    "3 - High-value stock options": 3
    }
    StockOptionLevel_label = st.selectbox("Stock Option Level", list(stock_option_levels.keys()))
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
    TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 3)
    WorkLifeBalance = st.slider("Work Life Balance (1=Bad to 4=Best)", 1, 4, 3)
    YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
    YearsInCurrentRole = st.slider("Years in Current Role", 0, 18, 3)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
    YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 4)

    data = {
        'Age': Age,
        'BusinessTravel': BusinessTravel,
        'DailyRate': DailyRate,
        'Department': Department,
        'DistanceFromHome': DistanceFromHome,
        'Education' : education_options[Education_label],
        'EducationField': EducationField,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'Gender': Gender,
        'HourlyRate': HourlyRate,
        'JobInvolvement': JobInvolvement,
        'JobLevel' : job_level_options[JobLevel_label],
        'JobRole': JobRole,
        'JobSatisfaction': JobSatisfaction,
        'MaritalStatus': MaritalStatus,
        'MonthlyIncome': MonthlyIncome,
        'MonthlyRate': MonthlyRate,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': OverTime,
        'PercentSalaryHike': PercentSalaryHike,
        'PerformanceRating': PerformanceRating,
        'RelationshipSatisfaction': RelationshipSatisfaction,
        'StockOptionLevel' : stock_option_levels[StockOptionLevel_label],
        'TotalWorkingYears': TotalWorkingYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager
    }

    return pd.DataFrame([data])

## Get input
input_df = user_input_features()

# Show input summary
st.subheader("📝 Input Summary")
st.dataframe(input_df)

# Predict button
if st.button("Predict"):
    # Encode using stored encoders
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            val = input_df.at[0, col]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                st.error(f"Unknown value '{val}' for column '{col}'. Expected one of: {list(le.classes_)}")
                st.stop()

    # Convert to float for model input
    input_df = input_df.astype(float)

    # Prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    attrition_probability = proba[1] * 100

    # Output
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.markdown("### 🔴 **Attrition Risk Detected**")
        st.warning("The employee is **likely to leave**.")
    else:
        st.markdown("### 🟢 **Low Attrition Risk**")
        st.success("The employee is **likely to stay**.")

    # Show probability as a metric and progress bar
    st.metric(label="Probability of Leaving", value=f"{attrition_probability:.2f}%")
    st.progress(int(attrition_probability))

    # Optional feedback
    if attrition_probability > 75:
        st.error("⚠️ Very High Risk - Immediate intervention may be needed.")
    elif attrition_probability > 50:
        st.warning("🔶 Moderate Risk - Consider engaging the employee.")
    else:
        st.success("✅ Low Risk - Retention outlook is good.")

