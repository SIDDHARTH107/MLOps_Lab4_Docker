"""
dashboard.py — Streamlit Dashboard for Loan Default Prediction

WHAT THIS DOES:
- Shows a nice web form where users can enter customer details
- When they click "Predict", it sends the data to the Flask API
- Displays the result with visual indicators (colors, gauges, etc.)

Think of it like:
- The WAITER in a restaurant
- Takes the customer's order (form input)
- Walks to the kitchen (calls Flask API)
- Brings back the dish (shows prediction result)

This runs as a SEPARATE Docker container from the Flask API.
They communicate over the internal Docker network.
"""

import streamlit as st
import requests
import json

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="centered"
)

# ============================================================
# Header Section
# ============================================================
st.title("🏦 Loan Default Prediction")
st.markdown("Enter customer details below to predict whether they will default on their loan.")
st.markdown("---")

# ============================================================
# API URL Configuration
# ============================================================
# Inside Docker Compose, containers talk to each other using service names
# "api" is the service name we'll define in docker-compose.yml
# Outside Docker (local testing), use localhost

import os
API_URL = os.environ.get("API_URL", "http://localhost:5000")

# ============================================================
# Input Form
# ============================================================
# Streamlit makes it super easy to create forms
# Each st.number_input / st.slider creates an interactive widget

st.subheader("📝 Customer Information")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "💰 Annual Income ($)",
        min_value=20000,
        max_value=150000,
        value=75000,
        step=5000,
        help="Customer's yearly income before taxes"
    )

    loan_amount = st.number_input(
        "💳 Loan Amount ($)",
        min_value=1000,
        max_value=50000,
        value=15000,
        step=1000,
        help="How much the customer wants to borrow"
    )

    credit_score = st.slider(
        "📊 Credit Score",
        min_value=300,
        max_value=850,
        value=680,
        help="300 = very poor, 850 = excellent"
    )

    months_employed = st.number_input(
        "🏢 Months Employed",
        min_value=0,
        max_value=360,
        value=48,
        help="How long at their current job"
    )

with col2:
    num_credit_lines = st.number_input(
        "📑 Number of Credit Lines",
        min_value=0,
        max_value=20,
        value=5,
        help="How many open credit accounts"
    )

    interest_rate = st.number_input(
        "📈 Interest Rate (%)",
        min_value=5.0,
        max_value=25.0,
        value=12.5,
        step=0.5,
        help="The loan's interest rate"
    )

    loan_term = st.selectbox(
        "📅 Loan Term (months)",
        options=[12, 24, 36, 48, 60],
        index=2,  # Default to 36 months
        help="How many months to repay"
    )

    dti_ratio = st.slider(
        "⚖️ Debt-to-Income Ratio",
        min_value=0.05,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="What fraction of income goes to debt payments"
    )

st.markdown("---")

# ============================================================
# Prediction Button
# ============================================================
if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):

    # Build the payload (data to send to the API)
    payload = {
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "months_employed": months_employed,
        "num_credit_lines": num_credit_lines,
        "interest_rate": interest_rate,
        "loan_term": loan_term,
        "dti_ratio": dti_ratio,
    }

    try:
        # Call the Flask API
        with st.spinner("Analyzing customer profile..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )

        if response.status_code == 200:
            result = response.json()

            # ============================================================
            # Display Results
            # ============================================================
            st.markdown("---")
            st.subheader("📋 Prediction Result")

            # Show the prediction with color coding
            if result["prediction"] == 1:
                st.error(f"⚠️ **HIGH RISK — Likely to Default**")
            else:
                st.success(f"✅ **LOW RISK — Unlikely to Default**")

            # Show probabilities in columns
            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                no_default_prob = result["confidence"]["no_default_probability"]
                st.metric(
                    label="No Default Probability",
                    value=f"{no_default_prob:.1%}"
                )

            with prob_col2:
                default_prob = result["confidence"]["default_probability"]
                st.metric(
                    label="Default Probability",
                    value=f"{default_prob:.1%}"
                )

            # Visual progress bar for default risk
            st.markdown("**Default Risk Level:**")
            st.progress(result["confidence"]["default_probability"])

            # Show the raw input sent to the API (for transparency)
            with st.expander("🔎 View raw API request & response"):
                st.json({"request": payload, "response": result})

        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the prediction API. "
            "Make sure the Flask API is running on " + API_URL
        )
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ============================================================
# Footer / Info Section
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        Built with Flask + Streamlit + Docker | MLOps Lab 4 | Northeastern University
    </div>
    """,
    unsafe_allow_html=True
)
