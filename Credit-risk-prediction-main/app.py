import streamlit as st
import joblib
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

st.set_page_config(
    page_title="AI Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

st.title("💳 AI Credit Risk Scoring System")
st.write("Enter customer information to predict the probability of default.")

model_path = os.path.join("model", "xgboost_credit_model.joblib")
model = joblib.load(model_path)

explainer = shap.TreeExplainer(model)

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    util_rate = st.number_input(
        "Credit Utilization Rate",
        min_value=0.0,
        value=0.50
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        value=0.30
    )

    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        value=0
    )

with col2:

    open_loans = st.number_input(
        "Open Credit Lines",
        min_value=0,
        value=8
    )

    real_estate_loans = st.number_input(
        "Real Estate Loans",
        min_value=0,
        value=1
    )

    late_30_59 = st.number_input(
        "Late Payments (30-59 Days)",
        min_value=0,
        value=0
    )

    late_60_89 = st.number_input(
        "Late Payments (60-89 Days)",
        min_value=0,
        value=0
    )

    late_90_plus = st.number_input(
        "Late Payments (90+ Days)",
        min_value=0,
        value=0
    )

income_is_missing = st.checkbox(
    "Income Missing",
    value=False
)

def create_pdf(probability, risk_level,Recommendation):

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 760, "Credit Risk Prediction Report")

    pdf.setFont("Helvetica", 12)

    pdf.drawString(50,720,f"Probability of Default: {probability*100:.2f}%")
    pdf.drawString(50,700,f"Risk Level: {risk_level}")

    pdf.drawString(50,590,"Generated using AI Credit Risk Prediction System")

    pdf.drawString(50, 660, "Recommendation:")
    pdf.drawString(70, 640, recommendation)
    pdf.save()

    buffer.seek(0)

    return buffer

predict = st.button("🔍 Predict Risk")

if predict:

    input_data = pd.DataFrame({
        "util_rate": [util_rate],
        "age": [age],
        "late_30_59": [late_30_59],
        "debt_ratio": [debt_ratio],
        "income": [income],
        "open_loans": [open_loans],
        "late_90_plus": [late_90_plus],
        "real_estate_loans": [real_estate_loans],
        "late_60_89": [late_60_89],
        "dependents": [dependents],
        "income_is_missing": [int(income_is_missing)]
    })

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Probability of Default",
        f"{probability*100:.2f}%"
    )
    st.progress(float(probability))

    if probability < 0.40:
        risk_level = "Low Risk"
        recommendation = "Customer appears financially stable. Loan approval is recommended."
        st.success("🟢 Low Risk")

    elif probability < 0.70:
        risk_level = "Medium Risk"
        recommendation = "Further financial review is recommended before approving the loan."
        st.warning("🟠 Medium Risk")

    else:
        risk_level = "High Risk"
        recommendation = "Further financial review is recommended before approving the loan."
        st.error("🔴 High Risk")

    st.subheader("📊 Why This Prediction Happened")

    shap_values = explainer(input_data)

    fig, ax = plt.subplots(figsize=(7,4))

    shap.plots.waterfall(
        shap_values[0],
        max_display=8,
        show=False
    )

    st.pyplot(fig)

    st.subheader("💡 Recommendation")

    if probability < 0.4:
       st.success("""
       The customer shows a low probability of default.
       • Stable financial profile
       • No significant late-payment history
       • Loan approval is recommended.
       """)

    elif probability < 0.7:
        st.warning("""
           The customer presents a moderate level of risk.
           • Additional verification is recommended.
           • Consider requesting supporting financial documents.
           """)

    else:
        st.error("""
        The customer has a high probability of default.
        • Multiple risk indicators were detected.
        • Loan approval is not recommended without further review.
        """)
    
    pdf = create_pdf(probability, risk_level,recommendation)

    st.download_button(
        label="📄 Download Prediction Report",
        data=pdf,
        file_name="credit_risk_report.pdf",
        mime="application/pdf"
    )