import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../models/best_model.pkl")

st.title("Customer Churn Prediction")
st.write(
    "Enter customer information to predict whether the customer is likely to churn."
)

# ---------- Helper ----------
# بدل ما نعرض للمستخدم 0 و 1 (اللي مش مفهومة)، بنعرضله "Yes / No"
# وبعدين نحولها لـ 0/1 جوّه الكود بس، عشان الموديل يفهمها
YES_NO = ["No", "Yes"]


def yes_no_to_binary(value: str) -> int:
    return 1 if value == "Yes" else 0


# ================= Inputs =================
# ----- Binary Features (Yes / No بدل 0 / 1) -----
gender = st.selectbox("Gender", ["Female", "Male"])
gender_val = 1 if gender == "Male" else 0

senior = st.selectbox("Senior Citizen", YES_NO)
partner = st.selectbox("Partner", YES_NO)
dependents = st.selectbox("Dependents", YES_NO)
phone_service = st.selectbox("Phone Service", YES_NO)
multiple_lines = st.selectbox("Multiple Lines", YES_NO)
online_security = st.selectbox("Online Security", YES_NO)
online_backup = st.selectbox("Online Backup", YES_NO)
device_protection = st.selectbox("Device Protection", YES_NO)
tech_support = st.selectbox("Tech Support", YES_NO)
streaming_tv = st.selectbox("Streaming TV", YES_NO)
streaming_movies = st.selectbox("Streaming Movies", YES_NO)
paperless = st.selectbox("Paperless Billing", YES_NO)
high_charge = st.selectbox("High Monthly Charge", YES_NO)
loyal_customer = st.selectbox("Loyal Customer", YES_NO)

# ----- Numeric Features -----
tenure = st.number_input("Tenure Months", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)
cltv = st.number_input("CLTV", min_value=0.0)
average = st.number_input("Average Charges", min_value=0.0)
cltv_month = st.number_input("CLTV per Month", min_value=0.0)

# ----- Contract -----
contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

# ----- Payment Method -----
payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# ----- Internet Service -----
internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# ----- Predict Button -----
predict = st.button("Predict")

# ================= Step 6 =================
# How can user selections be converted into the same feature format used
# during model training?
#
# The trained model expects One-Hot Encoded features, not text values.
# Therefore, we must convert the user's selections into the same feature
# format before making predictions.

if predict:

    # ---- Convert binary Yes/No selections to 0/1 ----
    senior_val = yes_no_to_binary(senior)
    partner_val = yes_no_to_binary(partner)
    dependents_val = yes_no_to_binary(dependents)
    phone_service_val = yes_no_to_binary(phone_service)
    multiple_lines_val = yes_no_to_binary(multiple_lines)
    online_security_val = yes_no_to_binary(online_security)
    online_backup_val = yes_no_to_binary(online_backup)
    device_protection_val = yes_no_to_binary(device_protection)
    tech_support_val = yes_no_to_binary(tech_support)
    streaming_tv_val = yes_no_to_binary(streaming_tv)
    streaming_movies_val = yes_no_to_binary(streaming_movies)
    paperless_val = yes_no_to_binary(paperless)
    high_charge_val = yes_no_to_binary(high_charge)
    loyal_customer_val = yes_no_to_binary(loyal_customer)

    # ---- Contract (One-Hot) ----
    contract_month = 0
    contract_one = 0
    contract_two = 0
    if contract == "Month-to-month":
        contract_month = 1
    elif contract == "One year":
        contract_one = 1
    else:
        contract_two = 1

    # ---- Payment Method (One-Hot) ----
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

    # ---- Internet Service (One-Hot) ----
    # ملحوظة: ده الجزء اللي كان فيه الخطأ الأصلي (indentation) وخلى الكود
    # يقع. اتصلح دلوقتي ويشتغل بشكل مستقل عن جزء الـ Payment.
    internet_dsl = 0
    internet_fiber = 0
    internet_no = 0
    if internet == "DSL":
        internet_dsl = 1
    elif internet == "Fiber optic":
        internet_fiber = 1
    else:
        internet_no = 1

    # خلصنا الجزء الصعب.
    # إحنا دلوقتي حولنا مثلاً:
    #   Contract = "One year"
    # إلى:
    #   Contract_Month-to-month = 0
    #   Contract_One year       = 1
    #   Contract_Two year       = 0
    # وده بالظبط نفس اللي عملناه في الـ Feature Engineering وقت التدريب.

    input_data = pd.DataFrame({
        "Gender": [gender_val],
        "Senior Citizen": [senior_val],
        "Partner": [partner_val],
        "Dependents": [dependents_val],
        "Tenure Months": [tenure],
        "Phone Service": [phone_service_val],
        "Multiple Lines": [multiple_lines_val],
        "Online Security": [online_security_val],
        "Online Backup": [online_backup_val],
        "Device Protection": [device_protection_val],
        "Tech Support": [tech_support_val],
        "Streaming TV": [streaming_tv_val],
        "Streaming Movies": [streaming_movies_val],
        "Paperless Billing": [paperless_val],
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
        "High Monthly Charge": [high_charge_val],
        "Loyal Customer": [loyal_customer_val],
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
