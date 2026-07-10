import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../models/best_model.pkl")

st.title("Customer Churn Prediction")
st.write(
    "Enter customer information to predict whether the customer is likely to churn."
)
#Inputs
#أولًا الـ Binary Features

gender = st.selectbox(
    "Gender",
    [0, 1]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    [0, 1]
)

dependents = st.selectbox(
    "Dependents",
    [0, 1]
)

phone_service = st.selectbox(
    "Phone Service",
    [0, 1]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    [0, 1]
)

online_security = st.selectbox(
    "Online Security",
    [0, 1]
)

online_backup = st.selectbox(
    "Online Backup",
    [0, 1]
)

device_protection = st.selectbox(
    "Device Protection",
    [0, 1]
)

tech_support = st.selectbox(
    "Tech Support",
    [0, 1]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    [0, 1]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    [0, 1]
)

paperless = st.selectbox(
    "Paperless Billing",
    [0, 1]
)

high_charge = st.selectbox(
    "High Monthly Charge",
    [0, 1]
)

loyal_customer = st.selectbox(
    "Loyal Customer",
    [0, 1]
)

#Numeric Features
tenure = st.number_input(
    "Tenure Months",
    min_value=0
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0
)

cltv = st.number_input(
    "CLTV",
    min_value=0.0
)

average = st.number_input(
    "Average Charges",
    min_value=0.0
)

cltv_month = st.number_input(
    "CLTV per Month",
    min_value=0.0
)


#Contract

contract = st.selectbox(

    "Contract",

    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

#Payment Method
payment = st.selectbox(

    "Payment Method",

    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

#Internet Service
internet = st.selectbox(

    "Internet Service",

    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

#predicton button
predict = st.button(
    "Predict"
)


#Step 6 - Convert User Input to Machine Learning Features


#How can user selections be converted into the same feature format used during model training?
# Why are we asking this?

#The trained model expects One-Hot Encoded features, not text values. Therefore, we must convert the user's
#selections into the same feature format before making predictions


if predict:
        contract_month = 0
        contract_one = 0
        contract_two = 0
        if contract == "Month-to-month":
         contract_month = 1

        elif contract == "One year":
         contract_one = 1

        else:
         contract_two = 1
        payment_bank = 0
        payment_credit = 0
        payment_electronic = 0
        payment_mailed = 0
        if payment == "Bank transfer (automatic)":
         payment_bank = 1

        elif payment == "Credit card (automatic)":
         payment_credit = 1

        elif payment == "Electronic check":
         payment_electronic = 1

        else:
         payment_mailed = 1
         internet_dsl = 0
        internet_fiber = 0
        internet_no = 0
        if internet == "DSL":
         internet_dsl = 1

        elif internet == "Fiber optic":
         internet_fiber = 1

        else:
         internet_no = 1


##خلصنا الجزء الصعب.

#إحنا دلوقتى حولنا:

#One year

#إلى

#Contract_Month-to-month = 0
#Contract_One year = 1
#Contract_Two year = 0

#وده بالظبط نفس اللى عملناه فى الـ Feature Engineering.

input_data = pd.DataFrame({

        "Gender": [gender],
        "Senior Citizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure Months": [tenure],
        "Phone Service": [phone_service],
        "Multiple Lines": [multiple_lines],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Paperless Billing": [paperless],
        "Monthly Charges": [monthly],
        "Total Charges": [total],
        "CLTV": [cltv],

        "Contract_Month-to-month": [contract_month],
        "Contract_One year": [contract_one],
        "Contract_Two year": [contract_two],

        "Payment Method_Bank transfer (automatic)": [payment_bank],
        "Payment Method_Credit card (automatic)": [payment_credit],
        "Payment Method_Electronic check": [payment_electronic],
        "Payment Method_Mailed check": [payment_mailed],

        "Internet Service_DSL": [internet_dsl],
        "Internet Service_Fiber optic": [internet_fiber],
        "Internet Service_No": [internet_no],

        "Average Charges": [average],
        "CLTV per Month": [cltv_month],
        "High Monthly Charge": [high_charge],
        "Loyal Customer": [loyal_customer]

    })



prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

st.subheader("Prediction Result")

if prediction[0] == 1:

        st.error("⚠️ Customer is likely to Churn")

else:


   st.success("✅ Customer is not likely to Churn")
   st.write(
        "Prediction Probability:",
        round(probability[0][1] * 100, 2),
        "%"
    )