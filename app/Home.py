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

st.title("Fraud Prediction")
col1,col2=st.columns(2)
with col1:
    with st.container(border=True):
            distance_from_home = st.number_input("Distance from Home (KM)", min_value=0.0, value=0.0,step=20.0)
            with st.expander(label='expand for details..'):
                st.write("This input allows one to input the distance of transaction from the home address (Probability of fraud increases with distance)")
            distance_from_last_transaction = st.number_input("Distance from Last Transaction (KM)", min_value=0.0, value=2.0,step=10.0)
            with st.expander(label='expand for details..'):
                st.write("This input allows one to input the distance of transaction from the last recorded transaction location (Probability of fraud increases with distance)")
            ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=1.0, value=1.0,step=1.0)
            with st.expander(label='expand for details..'):
                st.write("This input allows one to enter the ratio of transaction amount with that of the median transaction amount (Probability of fraud increases with increase in ratio)")
with col2:
    with st.container(border=True):
        repeat_retailer = st.checkbox("Repeat Retailer", [0, 1])
        with st.expander(label='expand for details..'):
            st.write("Was the transaction made with the same retailer?")
        used_chip = st.checkbox("Used Chip", [0, 1])
        with st.expander(label='expand for details..'):
            st.write("Was the transaction done with a credit card")
        used_pin_number = st.checkbox("Used PIN Number", [0, 1])
        with st.expander(label='expand for details..'):
            st.write("Was the pin number entered during the transaction?")
        online_order = st.checkbox("Online Order", [0, 1])
        with st.expander(label='expand for details..'):
            st.write("Was the transaction an online purchase?")

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

