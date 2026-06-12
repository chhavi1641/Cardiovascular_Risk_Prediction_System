import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model Files
# ==========================

model = joblib.load("cardio_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Cardiovascular Risk Prediction System")

st.markdown("""
Predict the likelihood of cardiovascular risk using demographic information,
vital signs, and laboratory measurements.
""")

# ==========================
# Sidebar Inputs
# ==========================

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 0, 100, 40)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

race = st.sidebar.selectbox(
    "Race",
    ["white", "black", "hawaiian", "other"]
)

ethnicity = st.sidebar.selectbox(
    "Ethnicity",
    ["nonhispanic", "other"]
)

income = st.sidebar.number_input(
    "Income",
    min_value=0,
    value=50000
)

healthcare_expenses = st.sidebar.number_input(
    "Healthcare Expenses",
    min_value=0,
    value=10000
)

# ==========================
# Clinical Measurements
# ==========================

st.subheader("Clinical Measurements")

col1, col2 = st.columns(2)

with col1:
    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    sbp = st.slider(
        "Systolic Blood Pressure",
        min_value=80,
        max_value=220,
        value=120
    )

    dbp = st.slider(
        "Diastolic Blood Pressure",
        min_value=40,
        max_value=140,
        value=80
    )

with col2:
    hr = st.slider(
        "Heart Rate",
        min_value=40,
        max_value=180,
        value=75
    )

    glucose = st.slider(
        "Glucose",
        min_value=50,
        max_value=400,
        value=100
    )

    hba1c = st.slider(
        "HbA1c",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

# ==========================
# Laboratory Results
# ==========================

st.subheader("Laboratory Results")

col3, col4 = st.columns(2)

with col3:

    cholesterol = st.slider(
        "Cholesterol",
        min_value=50,
        max_value=400,
        value=180
    )

    ldl = st.slider(
        "LDL",
        min_value=20,
        max_value=300,
        value=100
    )

with col4:

    hdl = st.slider(
        "HDL",
        min_value=10,
        max_value=120,
        value=50
    )

    creatinine = st.slider(
        "Creatinine",
        min_value=0.1,
        max_value=10.0,
        value=1.0
    )

# ==========================
# Prediction
# ==========================

if st.button("Predict Risk"):

    data = {
        "AGE": age,
        "INCOME": income,
        "HEALTHCARE_EXPENSES": healthcare_expenses,

        "AVG_BMI": bmi,
        "LATEST_BMI": bmi,

        "AVG_SBP": sbp,
        "LATEST_SBP": sbp,

        "AVG_DBP": dbp,
        "LATEST_DBP": dbp,

        "AVG_HR": hr,
        "LATEST_HR": hr,

        "AVG_GLUCOSE": glucose,
        "LATEST_GLUCOSE": glucose,

        "AVG_HBA1C": hba1c,
        "LATEST_HBA1C": hba1c,

        "AVG_CHOLESTEROL": cholesterol,
        "LATEST_CHOLESTEROL": cholesterol,

        "AVG_LDL": ldl,
        "LATEST_LDL": ldl,

        "AVG_HDL": hdl,
        "LATEST_HDL": hdl,

        "AVG_CREATININE": creatinine,
        "LATEST_CREATININE": creatinine,

        "GENDER_M": 1 if gender == "Male" else 0,

        "RACE_black": 1 if race == "black" else 0,
        "RACE_hawaiian": 1 if race == "hawaiian" else 0,
        "RACE_other": 1 if race == "other" else 0,
        "RACE_white": 1 if race == "white" else 0,

        "ETHNICITY_nonhispanic": 1 if ethnicity == "nonhispanic" else 0
    }

    # Create DataFrame
    input_df = pd.DataFrame([data])

    # Match training feature order
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Convert all columns to numeric
    input_df = input_df.astype(float)

    # Prediction
    risk_prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if risk_prob < 0.30:
        st.success(f"🟢 Low Risk ({risk_prob:.1%})")

    elif risk_prob < 0.60:
        st.warning(f"🟡 Moderate Risk ({risk_prob:.1%})")

    else:
        st.error(f"🔴 High Risk ({risk_prob:.1%})")

    st.metric(
        label="Risk Probability",
        value=f"{risk_prob:.1%}"
    )