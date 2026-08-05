# Customer Renewal Probability Predictor using best-performing features from model


import streamlit as st
import numpy as np
import pandas as pd
import pickle


@st.cache_resource
def load_artifacts():
    with open("churn_xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_v3.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder
model, encoder = load_artifacts()


st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal and the revenue at stake.")

tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5)
avg_session_length = st.number_input("Average Session Length (minutes)", min_value=0.0, max_value=500.0, value=30.0)
total_num_sessions = st.number_input("Total Number of Sessions", min_value=0, max_value=10000, value=100)
current_arr        = st.number_input("Current ARR ($)", min_value=0.0, max_value=10000.0, value=150.0)
product           = st.radio("Product",       ["Daily Fitness", "Healthy Meals", "Mindful Living", "Premium Health", "Wellness Tracker"])
income_level      = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
education         = st.radio("Education",     ["Graduate", "High School", "Other", "Post-Graduate"])
device_type       = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])

if st.button("Predict"):

    # Build categorical DataFrame — column names must match encoder exactly
    raw = pd.DataFrame([{
        'PRODUCT':      product,
        'EDUCATION':    education,
        'INCOME_LEVEL': income_level,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # All numeric features — must match training column order exactly
    # Features not collected from the user are held at fixed baseline values
    numeric_df = pd.DataFrame([{
        'AGE':                        0,
        'TECH_COMFORT_SCORE':         tech_comfort_score,
        'TOTAL_NUM_SESSIONS':         total_num_sessions,
        'ACTIVE_MONTHS':              0,
        'AVG_SESSION_LENGTH':         avg_session_length,
        'DAYS_SINCE_LAST_ACTIVITY':   0,
        'SESSION_GROWTH_RATIO':       0,
        'PRODUCTS_USED':              0,
        'DISTINCT_ACTIVITY_DAYS':     0,
        'SESSION_VARIABILITY':        0,
        'AVG_DAYS_TO_PAYMENT':        0,
        'LATE_PAYMENT_FLAG':          0,
        'NUM_SUBSCRIPTIONS_HIST':     0
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    # Expected Future ARR = what this customer is likely worth next year.
    # ARR at Risk = the gap between today's value and that expectation.
    expected_future_arr = current_arr * probability
    arr_at_risk = current_arr - expected_future_arr

    col1, col2, col3 = st.columns(3)
    col1.metric("Renewal Probability", f"{probability:.2f}")
    col2.metric("Expected Future ARR", f"${expected_future_arr:,.0f}")
    col3.metric("ARR at Risk", f"${arr_at_risk:,.0f}")

    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
