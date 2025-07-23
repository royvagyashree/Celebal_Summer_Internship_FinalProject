import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Creditworthiness Prediction App", page_icon="💳")

# Load the trained pipeline model
@st.cache_resource
def load_model():
    model = joblib.load("credit_model_pipeline.pkl")
    return model

model = load_model()

# Title
st.title("💳 Creditworthiness Prediction App")
st.markdown("Predict whether a person has **Good or Bad Credit** based on financial and demographic inputs.")

# Input form
with st.form("input_form"):
    st.markdown("### 🔍 Enter Customer Details")

    # Mappings
    status_checking_options = {'... < 0 DM': 'A11', '0 <= ... < 200 DM': 'A12', '... >= 200 DM': 'A13', 'No checking account': 'A14'}
    credit_history_options = {
        'No credits taken / all paid back': 'A30',
        'All at this bank paid back': 'A31',
        'Existing paid back till now': 'A32',
        'Delay in paying off': 'A33',
        'Critical / other credits exist': 'A34'
    }
    purpose_options = {
        'Car (new)': 'A40', 'Car (used)': 'A41', 'Furniture / equipment': 'A42',
        'Radio / television': 'A43', 'Domestic appliances': 'A44', 'Repairs': 'A45',
        'Education': 'A46', 'Retraining': 'A48', 'Business': 'A49', 'Others': 'A410'
    }
    savings_options = {'< 100 DM': 'A61', '100 <= ... < 500 DM': 'A62', '500 <= ... < 1000 DM': 'A63', '>= 1000 DM': 'A64', 'Unknown': 'A65'}
    employment_options = {'Unemployed': 'A71', '< 1 year': 'A72', '1 <= ... < 4 years': 'A73', '4 <= ... < 7 years': 'A74', '>= 7 years': 'A75'}
    personal_status_options = {
        'Male : divorced/separated': 'A91', 'Female : div/sep/married': 'A92',
        'Male : single': 'A93', 'Male : married/widowed': 'A94', 'Female : single': 'A95'
    }
    other_debtors_options = {'None': 'A101', 'Co-applicant': 'A102', 'Guarantor': 'A103'}
    property_options = {'Real estate': 'A121', 'Savings / life insurance': 'A122', 'Car or other': 'A123', 'Unknown': 'A124'}
    other_installment_options = {'Bank': 'A141', 'Stores': 'A142', 'None': 'A143'}
    housing_options = {'Rent': 'A151', 'Own': 'A152', 'For free': 'A153'}
    job_options = {
        'Unemployed / non-resident': 'A171', 'Unskilled - resident': 'A172',
        'Skilled employee / official': 'A173', 'Management / self-employed': 'A174'
    }
    telephone_options = {'None': 'A191', 'Yes, registered': 'A192'}
    foreign_worker_options = {'Yes': 'A201', 'No': 'A202'}

    # Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        status = st.selectbox("Status of Checking Account", list(status_checking_options.keys()))
        duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=12)
        credit_hist = st.selectbox("Credit History", list(credit_history_options.keys()))
        purpose = st.selectbox("Purpose", list(purpose_options.keys()))
        credit_amt = st.number_input("Credit Amount", min_value=100, max_value=10000, value=1000)
        savings = st.selectbox("Savings Account", list(savings_options.keys()))
        employment = st.selectbox("Employment Since", list(employment_options.keys()))

    with col2:
        installment_rate = st.number_input("Installment Rate (%)", min_value=1, max_value=10, value=2)
        personal_status = st.selectbox("Personal Status / Sex", list(personal_status_options.keys()))
        other_debtors = st.selectbox("Other Debtors", list(other_debtors_options.keys()))
        residence_since = st.slider("Residence Since (years)", min_value=1, max_value=10, value=2)
        property_type = st.selectbox("Property", list(property_options.keys()))
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        other_installments = st.selectbox("Other Installment Plans", list(other_installment_options.keys()))

    with col3:
        housing = st.selectbox("Housing", list(housing_options.keys()))
        existing_credits = st.slider("Existing Credits", 1, 4, 1)
        job = st.selectbox("Job Type", list(job_options.keys()))
        liable_people = st.slider("No. of People Liable", 1, 2, 1)
        telephone = st.selectbox("Telephone", list(telephone_options.keys()))
        foreign_worker = st.selectbox("Foreign Worker", list(foreign_worker_options.keys()))

    # Feature Engineering Helpers
    def get_age_group(age):
        if age < 26:
            return '18-25'
        elif age < 36:
            return '26-35'
        elif age < 51:
            return '36-50'
        else:
            return '50+'

    def get_installment_cat(rate):
        if rate <= 2:
            return 'Low'
        elif rate == 3:
            return 'Medium'
        else:
            return 'High'

    def get_duration_bucket(months):
        if months <= 12:
            return 'Short'
        elif months <= 24:
            return 'Medium'
        elif months <= 36:
            return 'Long'
        else:
            return 'Very Long'

    # Create DataFrame
    df = pd.DataFrame({
        'Status_Checking_Acc': [status_checking_options[status]],
        'Duration': [duration],
        'Credit_History': [credit_history_options[credit_hist]],
        'Purpose': [purpose_options[purpose]],
        'Credit_Amount': [credit_amt],
        'Savings_Account': [savings_options[savings]],
        'Employment_Since': [employment_options[employment]],
        'Installment_Rate': [installment_rate],
        'Personal_Status_Sex': [personal_status_options[personal_status]],
        'Other_Debtors': [other_debtors_options[other_debtors]],
        'Residence_Since': [residence_since],
        'Property': [property_options[property_type]],
        'Age': [age],
        'Other_Installment_Plans': [other_installment_options[other_installments]],
        'Housing': [housing_options[housing]],
        'Existing_Credits': [existing_credits],
        'Job': [job_options[job]],
        'Liable_People': [liable_people],
        'Telephone': [telephone_options[telephone]],
        'Foreign_Worker': [foreign_worker_options[foreign_worker]]
    })

    # Add engineered features
    df['Credit_to_Duration_Ratio'] = df['Credit_Amount'] / df['Duration']
    df['Age_Group'] = df['Age'].apply(get_age_group)
    df['Installment_Category'] = df['Installment_Rate'].apply(get_installment_cat)
    df['Duration_Bucket'] = df['Duration'].apply(get_duration_bucket)

    st.subheader("📋 Final Input DataFrame with Engineered Features:")
    st.dataframe(df)

    submitted = st.form_submit_button("Predict")

# Prediction block
if submitted:
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]

    st.subheader("🔮 Prediction Result")
    if prediction == 1:
        st.success(f"✅ The person is likely **creditworthy** (Good).")
    else:
        st.error(f"⚠️ The person is likely **not creditworthy** (Bad).")

    st.markdown(f"**Confidence:** Good: `{prediction_proba[1]*100:.2f}%`, Bad: `{prediction_proba[0]*100:.2f}%`")
