import joblib
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Visuals", layout="wide")

xgb_model = joblib.load("../models/xgb_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

@st.cache_data
def load_data(sample_frac=0.5, random_state=42):
    df = pd.read_csv("../data/cleaned_fraud_dataset.csv")
    df = df.sample(frac=sample_frac, random_state=random_state)
    return df

def annotate_bars(ax):
    for p in ax.patches:
        value = int(p.get_height())
        ax.annotate(f"{value:,}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=8, color="black", xytext=(0, 2),
                    textcoords="offset points")

@st.cache_resource
def fraud_distribution_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="fraud", data=df, palette="pastel", ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def distance_from_home_plot(df):
    fig, ax = plt.subplots()
    sns.barplot(x="fraud", y="distance_from_home", data=df, palette="pastel", ci=None, ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def distance_from_last_transaction_plot(df):
    fig, ax = plt.subplots()
    sns.barplot(x="fraud", y="distance_from_last_transaction", data=df, palette="pastel", ci=None, ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def ratio_to_median_purchase_plot(df):
    fig, ax = plt.subplots()
    sns.barplot(x="fraud", y="ratio_to_median_purchase_price", data=df, palette="pastel", ci=None, ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def repeat_retail_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="repeat_retailer", hue="fraud", data=df, palette="pastel", ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def online_order_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="online_order", hue="fraud", data=df, palette="pastel", ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def used_chip_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="used_chip", hue="fraud", data=df, palette="pastel", ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def used_pin_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="used_pin_number", hue="fraud", data=df, palette="pastel", ax=ax)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig


df = load_data()
st.write("""### Dataset Preview
         A peak at the Fraudulent Transaction dataset""", df.tail(50))

st.write("""#### Dataset Distribuition:
         - Fraud : 43,679
         - Non Fraud : 4,56,321

         note: Only half of the full Dataset is used for the visualisations
         """)

st.markdown("""
### Feature Explanations:
- **distance_from_home** : Distance in Kilometers from the cardholder’s home and the transaction point.
- **distance_from_last_transaction** : Distance in Kilometers between this transaction's location and the previous one.
- **ratio_to_median_purchase_price** : Ratio of transaction amount to median purchase price.
- **repeat_retailer** : Whether the purchase is from a retailer that was previously used.
- **used_chip** : Whether the transaction was completed using the card’s chip.
- **used_pin_number** : Whether a PIN number was used.
- **online_order** : Whether the transaction was done online.
- **fraud** : Target variable (1 = fraud, 0 = normal).
""")

st.divider()

def feature_importance_plot(xgb_model, feature_names):
    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color="orange")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")
    return fig

st.header("Data Visualisations")


row1 = st.columns(4)

with row1[0]:
    st.subheader("Fraudulent Transaction Distribution")
    st.pyplot(fraud_distribution_plot(df))
    st.markdown("The dataset is **highly imbalanced** with far fewer fraud cases.")

with row1[1]:
    st.subheader("Distance of Transaction from Home")
    st.pyplot(distance_from_home_plot(df))
    st.markdown("Fraudulent transactions tend to happen **further away from home** on average.")

with row1[2]:
    st.subheader("Distance of Transaction from last Location")
    st.pyplot(distance_from_last_transaction_plot(df))
    st.markdown("Fraudulent transactions tend to occur **farther from the last location** compared to normal transactions.")

with row1[3]:
    st.subheader("Ratio of Transaction to Median Purchase Price")
    st.pyplot(ratio_to_median_purchase_plot(df))
    st.markdown("Fraudulent transactions are often linked with a **higher deviation in purchase price ratios**.")



row2 = st.columns(4)

with row2[0]:
    st.subheader("Repeated Retailer Distribution")
    st.pyplot(repeat_retail_plot(df))
    st.markdown("Fraudulent transactions tend to occur **more frequently with repeat retailers**.")

with row2[1]:
    st.subheader("Credit Card Chip usage Distribution")
    st.pyplot(used_chip_plot(df))
    st.markdown("Fraudulent transactions are less likely to use the **chip authentication method**.")

with row2[2]:
    st.subheader("Used PIN Number Distribution")
    st.pyplot(used_pin_plot(df))
    st.markdown("Fraudulent transactions are less likely to involve a **PIN number**.")

with row2[3]:
    st.subheader("Transaction as Online Order Distribution")
    st.pyplot(online_order_plot(df))
    st.markdown("Fraudulent transactions tend to occur **more often in online purchases**.")

st.divider()
st.header("Model insights")
col1,col2=st.columns(2)
with col1:
    st.subheader("Model Feature Importances")
    st.pyplot(feature_importance_plot(xgb_model, feature_names))
with col2:
    st.markdown("""
### What it means
The Feature Importance plot gives us insight to how the Machine interprets the data and what it thinks are the biggest indicators of a fraud
- **ratio_to_median_purchase_price** is the strongest indicator of fraud.
- Online transactions and long distances from home are also highly suspicious.
- Secure authentication methods (PIN, card's chip) are less common in fraud.
- Location history and repeat retailers add minor predictive power.""")

st.markdown("## ROC-AUC curve")
rows1=st.columns(3)
with rows1[0]:
    st.image("../visuals/Roc(isof).png", caption="ROC curve for the Isolation forest model")
with rows1[1]:
    st.image("../visuals/Roc(lof).png", caption="ROC curve for the Local Outlier Factor model")
with rows1[2]:
    st.image("../visuals/Roc(xgb).png", caption="ROC curve for the XGBoost model")

col1,col2=st.columns(2)
with col2:
    st.markdown(""" ### Model Comparisons
    - **Isolation Forest (AUC ≈ 0.74):** Some fraud detection ability, but weak overall.
    - **Local Outlier Factor (AUC ≈ 0.74):** Similar performance, moderate accuracy.
    - **XGBoost (AUC = 1.0):** Top Performer, with best accuracy.
    """)
with col1:
    st.markdown("""
    ### What is ROC-AUC?
    - **ROC Curve:** Plots True Positive Rate vs False Positive Rate.
    - **AUC (Area Under Curve):** Higher is better (1 = perfect, 0.5 = random guess).
    - AUC helps measure how well the model separates fraud from normal.""")
