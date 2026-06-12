import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model Files
# ==========================

model = joblib.load("cardio_risk_model.pkl")
imputer = joblib.load("imputer.pkl")
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
st.markdown(
    """
    Predict the likelihood of cardiovascular risk using
    demographic information, vital signs, and laboratory measurements.
    """
)

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
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    sbp = st.slider("Systolic BP", 80, 220, 120)
    dbp = st.slider("Diastolic BP", 40, 140, 80)

with col2:
    hr = st.slider("Heart Rate", 40, 180, 75)
    glucose = st.slider("Glucose", 50, 400, 100)
    hba1c = st.slider("HbA1c", 3.0, 15.0, 5.5)

# ==========================
# Lab Values
# ==========================

st.subheader("Laboratory Results")

col3, col4 = st.columns(2)

with col3:
    cholesterol = st.slider(
        "Cholesterol",
        50,
        400,
        180
    )

    ldl = st.slider(
        "LDL",
        20,
        300,
        100
    )

with col4:
    hdl = st.slider(
        "HDL",
        10,
        120,
        50
    )

    creatinine = st.slider(
        "Creatinine",
        0.1,
        10.0,
        1.0
    )

# ==========================
# Prediction Button
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

        "ETHNICITY_nonhispanic":
            1 if ethnicity == "nonhispanic" else 0
    }

    input_df = pd.DataFrame([data])

    # Match training order
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    input_df = pd.DataFrame(
        imputer.transform(input_df),
        columns=input_df.columns
    )

    risk_prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if risk_prob < 0.30:
        st.success(
            f"Low Risk ({risk_prob:.1%})"
        )

    elif risk_prob < 0.60:
        st.warning(
            f"Moderate Risk ({risk_prob:.1%})"
        )

    else:
        st.error(
            f"High Risk ({risk_prob:.1%})"
        )

    st.metric(
        "Risk Probability",
        f"{risk_prob:.1%}"
    )