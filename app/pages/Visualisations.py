import joblib
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Visuals", layout="wide")

# Load model/scaler for consistency
xgb_model = joblib.load("../models/xgb_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

# =====================
# Helpers
# =====================

@st.cache_data
def load_data(sample_frac=0.5, random_state=42):
    df = pd.read_csv("../data/cleaned_fraud_dataset.csv")
    df = df.sample(frac=sample_frac, random_state=random_state)  # use 50% of data
    return df

def annotate_bars(ax):
    """Add labels on top of bars in seaborn barplot/countplot"""
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
    ax.set_title("Fraud vs Normal Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def distance_from_home_plot(df):
    fig, ax = plt.subplots()
    sns.barplot(x="fraud", y="distance_from_home", data=df, palette="pastel", ci=None, ax=ax)
    ax.set_title("Avg Distance from Home")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def distance_from_last_transaction_plot(df):
    fig, ax = plt.subplots()
    sns.barplot(x="fraud", y="distance_from_last_transaction", data=df, palette="pastel", ci=None, ax=ax)
    ax.set_title("Avg Distance from Last Transaction")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def ratio_to_median_purchase_plot(df):
    fig, ax = plt.subplots()
    sns.barplot(x="fraud", y="ratio_to_median_purchase_price", data=df, palette="pastel", ci=None, ax=ax)
    ax.set_title("Avg Ratio to Median Purchase Price")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def repeat_retail_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="repeat_retailer", hue="fraud", data=df, palette="pastel", ax=ax)
    ax.set_title("Repeat Retailer Distribution by Fraud")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def online_order_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="online_order", hue="fraud", data=df, palette="pastel", ax=ax)
    ax.set_title("Online Order Distribution by Fraud")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def used_chip_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="used_chip", hue="fraud", data=df, palette="pastel", ax=ax)
    ax.set_title("Chip Usage Distribution by Fraud")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig

@st.cache_resource
def used_pin_plot(df):
    fig, ax = plt.subplots()
    sns.countplot(x="used_pin_number", hue="fraud", data=df, palette="pastel", ax=ax)
    ax.set_title("PIN Number Usage by Fraud")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    annotate_bars(ax)
    return fig


df = load_data()
st.write("### Dataset Preview", df.head(10))

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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fraudulent Transaction Distribution")
    st.pyplot(fraud_distribution_plot(df))
    st.markdown("The dataset is **highly imbalanced** with far fewer fraud cases.")

    st.subheader("Distance of Transaction from Home")
    st.pyplot(distance_from_home_plot(df))
    st.markdown("Fraudulent transactions tend to happen **further away from home** on average.")

    st.subheader("Distance from Last Transaction")
    st.pyplot(distance_from_last_transaction_plot(df))
    st.markdown("Fraudulent transactions tend to occur **farther from the last location** compared to normal transactions.")

    st.subheader("Ratio to Median Purchase Price")
    st.pyplot(ratio_to_median_purchase_plot(df))
    st.markdown("Fraudulent transactions are often linked with a **higher deviation in purchase price ratios**.")

with col2:
    st.subheader("Repeated Retailer Distribution")
    st.pyplot(repeat_retail_plot(df))
    st.markdown("Fraudulent transactions tend to occur **more frequently with repeat retailers**.")

    st.subheader("Used Chip on Credit Card")
    st.pyplot(used_chip_plot(df))
    st.markdown("Fraudulent transactions are less likely to use the **chip authentication method**.")

    st.subheader("Used PIN Number Distribution")
    st.pyplot(used_pin_plot(df))
    st.markdown("Fraudulent transactions are less likely to involve a **PIN number**.")

    st.subheader("Online Order Distribution")
    st.pyplot(online_order_plot(df))
    st.markdown("Fraudulent transactions tend to occur **more often in online purchases**.")
