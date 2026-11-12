import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load("attrition_model.pkl")


bundle = load_model()
model = bundle["model"]

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

RATING = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
BALANCE = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}


def rating(label, labels=RATING):
    return st.select_slider(label, options=list(labels), value=3, format_func=labels.get)


with st.form("employee"):
    st.subheader("👤 Employee profile")
    c1, c2 = st.columns(2)
    with c1:
        Age = st.slider("Age", 18, 60, 30)
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        DistanceFromHome = st.slider("Distance from Home (km)", 1, 29, 10)
    with c2:
        JobRole = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"])
        OverTime = st.radio("Works Overtime?", ["No", "Yes"], horizontal=True)
        BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                      index=1, format_func=lambda v: v.replace("_", " "))

    st.subheader("📅 Career history")
    c1, c2 = st.columns(2)
    with c1:
        NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 9, 2)
        YearsInCurrentRole = st.slider("Years in Current Role", 0, 18, 3)
        TrainingTimesLastYear = st.slider("Training Sessions Last Year", 0, 6, 3)
    with c2:
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 4)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)

    st.subheader("😊 Satisfaction & engagement")
    c1, c2 = st.columns(2)
    with c1:
        JobSatisfaction = rating("Job Satisfaction")
        EnvironmentSatisfaction = rating("Environment Satisfaction")
        RelationshipSatisfaction = rating("Relationship Satisfaction")
    with c2:
        JobInvolvement = rating("Job Involvement")
        WorkLifeBalance = rating("Work Life Balance", BALANCE)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "OverTime": OverTime,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "BusinessTravel": BusinessTravel,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "JobSatisfaction": JobSatisfaction,
        "JobInvolvement": JobInvolvement,
        "WorkLifeBalance": WorkLifeBalance,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "Age": Age,
        "DistanceFromHome": DistanceFromHome,
        "NumCompaniesWorked": NumCompaniesWorked,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsWithCurrManager": YearsWithCurrManager,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
    }])[bundle["features"]]

    with st.expander("📝 Input Summary"):
        st.dataframe(input_df.T.astype(str).rename(columns={0: "Value"}))

    # The saved pipeline encodes categories and scales numbers itself.
    prediction = model.predict(input_df)[0]
    attrition_probability = model.predict_proba(input_df)[0][1] * 100

    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.markdown("### 🔴 **Attrition Risk Detected**")
        st.warning("The employee is **likely to leave**.")

        # 💡 Retention Recommendations
        recommendations = []
        if OverTime == "Yes":
            recommendations.append("🔁 Consider reducing excessive overtime or redistributing workload.")
        if JobSatisfaction <= 2:
            recommendations.append("💼 Conduct a satisfaction survey or 1-on-1 session to understand dissatisfaction.")
        if EnvironmentSatisfaction <= 2:
            recommendations.append("🏢 Review the work environment: workspace, tools, and team culture.")
        if WorkLifeBalance <= 2:
            recommendations.append("🏖️ Promote better work-life balance with flexible hours or remote options.")
        if BusinessTravel == "Travel_Frequently":
            recommendations.append("✈️ Look at reducing travel frequency or offering recovery time after trips.")
        if YearsSinceLastPromotion >= 3:
            recommendations.append("📈 Evaluate if the employee is due for a promotion or new responsibilities.")
        if RelationshipSatisfaction <= 2:
            recommendations.append("🤝 Encourage team bonding or assign a mentor to improve workplace relationships.")
        if TrainingTimesLastYear < 2:
            recommendations.append("📚 Offer skill development or training opportunities to boost engagement.")

        if recommendations:
            st.subheader("🛠️ Retention Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("No major concerns detected. Still, regular feedback and career discussions are helpful.")

    else:
        st.markdown("### 🟢 **Low Attrition Risk**")
        st.success("The employee is **likely to stay**.")

    # Show probability as a metric and progress bar
    st.metric(label="Probability of Leaving", value=f"{attrition_probability:.2f}%")
    st.progress(int(attrition_probability))

st.caption(f"Model: Logistic Regression on 16 key factors · "
           f"cross-validated accuracy {bundle['cv_accuracy']:.1%}")
