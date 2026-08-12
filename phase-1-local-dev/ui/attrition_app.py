import streamlit as st
import requests

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("Employee Attrition Prediction")

import os

# ── In GKE, the Kubernetes Service DNS name resolves this automatically.
#    Locally/Cloud Shell, override with: export API_URL=http://localhost:8080/predict
API_URL = os.environ.get("API_URL", "http://attrition-api:8080/predict")

col1, col2 = st.columns(2)

with col1:
    years_at_company = st.number_input(
        "Years at Company", min_value=0.0, max_value=50.0, value=5.0,
        help="Years worked in current role."
    )
    no_of_promotions = st.number_input("Number of Promotions", min_value=0, max_value=20, value=1)
    overtime_label = st.selectbox("Overtime", ["No", "Yes"])
    performance_label = st.selectbox("Performance Rating", ["Low", "Below Avg", "Avg", "High"])
    job_level_label = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
    company_reputation_label = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])
    opportunities = st.number_input(
        "Opportunities (0-2)", min_value=0, max_value=2, value=1,
        help="0: Low | 1: Medium | 2: High"
    )
    age_group_label = st.selectbox("Age Group", ["18-25", "25-35", "35-45", "45-60", "60-65"])

with col2:
    company_tenure = st.number_input(
        "Company Tenure", min_value=0.0, max_value=50.0, value=8.0,
        help="Total years worked in the company."
    )
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
    remote_work_label = st.selectbox("Remote Work", ["No", "Yes"])
    edu_level_label = st.selectbox(
        "Education Level", ["School", "Bachelors", "Masters", "Associate", "PhD"]
    )
    company_size_label = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    overall_satisfaction = st.number_input(
        "Overall Satisfaction (1-4)", min_value=1, max_value=4, value=3,
        help="1: Low | 2: Medium | 3: High | 4: Very High"
    )
    annual_income_label = st.selectbox(
        "Annual Income", ["Under 2.4L", "2.4L-4.2L", "4.2L-6L", "6L-20L", "Above 20L"]
    )

# ── Label -> numeric mappings (must match your training encoding) ──
overtime_map = {"No": 0, "Yes": 1}
performance_map = {"Low": 1, "Below Avg": 2, "Avg": 3, "High": 4}
edu_level_map = {"School": 1, "Bachelors": 2, "Masters": 3, "Associate": 4, "PhD": 5}
job_level_map = {"Entry": 1, "Mid": 2, "Senior": 3}
company_size_map = {"Small": 1, "Medium": 2, "Large": 3}
remote_work_map = {"No": 0, "Yes": 1}
company_reputation_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
annual_income_map = {"Under 2.4L": 0, "2.4L-4.2L": 1, "4.2L-6L": 2, "6L-20L": 3, "Above 20L": 4}
age_group_map = {"18-25": 1, "25-35": 2, "35-45": 3, "45-60": 4, "60-65": 5}

payload = {
    "years_at_company": years_at_company,
    "performance_rating": performance_map[performance_label],
    "no_of_promotions": no_of_promotions,
    "overtime": overtime_map[overtime_label],
    "edu_level": edu_level_map[edu_level_label],
    "no_of_dependents": no_of_dependents,
    "job_level": job_level_map[job_level_label],
    "company_size": company_size_map[company_size_label],
    "company_tenure": company_tenure,
    "remote_work": remote_work_map[remote_work_label],
    "company_reputation": company_reputation_map[company_reputation_label],
    "overall_satisfaction": overall_satisfaction,
    "opportunities": opportunities,
    "annual_income": annual_income_map[annual_income_label],
    "age_group": age_group_map[age_group_label],
}

st.divider()

if st.button("Predict", type="primary"):
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        risk_colors = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "VERY_LOW": "🟢"}
        icon = risk_colors.get(result.get("risk", ""), "")

        st.subheader("Result")
        if result["prediction"] == 1:
            st.error(f"{icon} Likely to LEAVE — risk: {result['risk']}")
        else:
            st.success(f"{icon} Likely to STAY — risk: {result['risk']}")

        c1, c2 = st.columns(2)
        c1.metric("P(Leave)", f"{result['p_leave']:.2%}")
        c2.metric("P(Stay)", f"{result['p_stay']:.2%}")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the prediction API: {e}")

with st.expander("Raw request payload"):
    st.json(payload)
