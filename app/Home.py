import joblib
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Loading Saved XGB model, Scaling Function and Feature order.
xgb_model = joblib.load("../models/xgb_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
feature_names = joblib.load("../models/feature_names.pkl")
st.set_page_config(page_title="Fraud Detection Visuals", layout="wide")
st.title("Fraud Prediction")
col1,col2=st.columns(2)
with col1:
    with st.container(border=True):
            distance_from_home = st.number_input("Distance from Home (KM)", min_value=0.0, value=0.0,step=20.0)
            with st.expander(label='More Details'):
                 st.write("""
        This is the distance between the cardholder's home address and
        the location of the transaction.
        - Legitimate purchases usually happen closer to home.
        - A much larger distance could indicate suspicious activity,
          especially if the cardholder is unlikely to be at that location.
        """)
            distance_from_last_transaction = st.number_input("Distance from Last Transaction (KM)", min_value=0.0, value=2.0,step=10.0)
            with st.expander(label='More Details'):
               st.write("""
        This measures how far the current transaction is from the previous one.
        - If the cardholder made a purchase nearby just minutes ago,
          then another transaction hundreds of kilometers away is highly unusual.
        - Larger distances in a short timeframe increase the chance of fraud.
        """)
            ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=1.0, value=1.0,step=1.0)
            with st.expander(label='More Details'):
                  st.write("""
        This is the ratio of the transaction amount compared to the cardholder’s
        typical (median) purchase amount.
        - If the ratio is close to 1, it means the transaction amount is normal.
        - If the ratio is very high, it could mean an unusually large purchase,
          which is often associated with fraud attempts.
        """)
with col2:
    with st.container(border=True):
        repeat_retailer = st.checkbox("Repeat Retailer", value=False)
        with st.expander(label='More Details'):
            st.write("""
            Indicates whether the transaction is with a retailer the cardholder
            has used before.
            - Regular purchases from the same retailer are usually safe.
            - Fraudulent transactions are more likely at new or unfamiliar retailers.
            """)
        used_chip = st.checkbox("Used Chip", value=False)
        with st.expander(label='More Details'):
            st.write("""
            Indicates whether the transaction was completed using the card’s chip.
            - Chip transactions are more secure since they use dynamic data.
            - Fraudulent transactions are often attempted with methods that bypass
              the chip.
            """)
        used_pin_number = st.checkbox("Used PIN Number", value=False)
        with st.expander(label='More Details'):
             st.write("""
           Indicates whether a PIN number was entered to authorize the transaction.
            - Transactions with PIN are generally safer because they require
              additional authentication.
            - Fraudsters often try to avoid PIN-based transactions.
            """)
        online_order = st.checkbox("Online Order", value=False)
        with st.expander(label='More Details'):
           st.write("""
            Identifies if the transaction was made online.
            - Online purchases are more vulnerable to fraud since no physical
              card or PIN is required.
            """)

if st.button("Predict"):
    input_data = {
        "distance_from_home": distance_from_home,
        "distance_from_last_transaction": distance_from_last_transaction,
        "ratio_to_median_purchase_price": ratio_to_median_purchase_price,
        "repeat_retailer": repeat_retailer,
        "used_chip": used_chip,
        "used_pin_number": used_pin_number,
        "online_order": online_order,
    }

    input_df = pd.DataFrame([input_data])

    numeric_features = ['distance_from_home',
                    'distance_from_last_transaction',
                    'ratio_to_median_purchase_price']
    X_num = scaler.transform(input_df[numeric_features])


    categorical_features = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    X_cat = input_df[categorical_features].values

    X_final = np.hstack([X_num, X_cat])
    fraud_prob = xgb_model.predict_proba(X_final)[0, 1]
    prediction = xgb_model.predict(X_final)[0]


    st.subheader("Prediction Result")
    st.markdown(f"Fraud Probability: {fraud_prob:.2%}")
    if prediction == 1:
        st.error("Potential Fraudulent Transaction Detected!")
    else:
        st.success("Normal Transaction")

