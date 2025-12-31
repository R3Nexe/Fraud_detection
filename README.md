# Fraud Detection System with Visual Analytics

## [Live Demo](https://fraudetection-app.streamlit.app/)

## Overview
This project presents a fraud detection system built using **XGBoost** and deployed as a **Streamlit web application**.  
The goal is to identify fraudulent credit card transactions based on key behavioral and transactional features.  
The system provides:
- Interactive visualizations to understand the dataset.
- A trained fraud detection model with feature importance analysis.
- A prediction interface where users can test custom inputs.
- Model evaluation through ROC curves and classification reports.

---

## Dataset
dataset used: [credit-card-fraud](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)

- **distance_from_home**: Distance (in kilometers) between the cardholder’s home and the transaction location.
- **distance_from_last_transaction**: Distance between the current transaction location and the previous one.
- **ratio_to_median_purchase_price**: Ratio of the transaction amount to the cardholder’s median purchase price.
- **repeat_retailer**: Whether the transaction was made with a retailer previously used.
- **used_chip**: Whether the transaction was completed using the card’s chip.
- **used_pin_number**: Whether a PIN number was entered.
- **online_order**: Whether the transaction was made online.
- **fraud**: Target variable (1 = fraud, 0 = normal).

---

## Methodology

1. **Data Preprocessing**
   - Cleaning and feature scaling.
   - Sampling strategies to handle class imbalance.
   - Encoding categorical variables.

2. **Model Training**
   - Primary model: **XGBoost Classifier**.
   - Other models trained for comparison (e.g., Logistic Regression, Random Forest).
   - Hyperparameter tuning performed with cross-validation.

3. **Evaluation Metrics**
   - **ROC-AUC Score**: To measure model discrimination power.
   - **Precision, Recall, F1-Score**: To evaluate performance on imbalanced data.
   - **Confusion Matrix**: To analyse false positives and false negatives.

---

## Visualizations
The Streamlit app provides a range of visual insights:

- **Fraudulent Transaction Distribution**  
  Shows the class imbalance between fraud and non-fraud cases.

- **Distance-Based Analysis**  
  - Fraud vs. non-fraud in relation to `distance_from_home` and `distance_from_last_transaction`.

- **Purchase Behavior**  
  - `ratio_to_median_purchase_price` distribution for fraud vs. normal transactions.

- **Categorical Features**  
  - Fraud correlation with `repeat_retailer`, `used_chip`, `used_pin_number`, and `online_order`.

- **Feature Importance**  
  Visual representation of which features are most influential for the XGBoost model.

---

## Streamlit Application

### Features
1. **Dataset Exploration**  
   Preview of the dataset and its key variables.

2. **Visual Analysis**  
   Interactive plots for better understanding of fraud patterns.

3. **Model Insights**  
   - Feature importance charts.
   - ROC curves and classification reports.

4. **Fraud Prediction Interface**  
   - User-friendly input forms for transaction details.  
   - Model outputs probability of fraud in real time.

---

## Installation and Usage

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/R3Nexe/Fraud_detection.git
cd fraud-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

