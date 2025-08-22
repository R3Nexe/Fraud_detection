#  Fraud Detection Web App

## [Live Demo](https://fraudetection-app.streamlit.app/)

An end-to-end fraud detection system built using XGBoost, Scikit-Learn, Streamlit, Seaborn, and Matplotlib.  
I developed this Machine Learning project during my [Elevate Labs](https://www.linkedin.com/search/results/all/?heroEntityKey=urn%3Ali%3Aorganization%3A106312219&keywords=Elevate%20Labs&origin=GLOBAL_SEARCH_HEADER&sid=~2%2C) Internship, focusing on preprocessing simulated financial transaction data, training and evaluating machine learning models, and deploying them as an interactive web app with real-time predictions and visual insights.

---

##  Project Overview
The aim of this project was to build a machine learning model that predicts fraudulent transactions and deploy it as a user friendly Streamlit web application.

The app allows:
-  Single transaction fraud prediction.
-  Dataset visualisations with explanations of key fraud patterns.



### Web App Preview
#### Preview of interactive home page
<img width="1683" height="968" alt="ScreenShot of Home page(Fraud Detected)" src="https://github.com/user-attachments/assets/0d90fdcd-e303-47db-b8c5-1842817dd0a4" />
<img width="1689" height="952" alt="ScreenShot of Normal Transaction" src="https://github.com/user-attachments/assets/36146824-834e-48ae-968a-a3cda91dc05c" />

#### Preview of Data visualisations and model insights
<img width="1704" height="970" alt="ScreenShot of Data Visualisations" src="https://github.com/user-attachments/assets/5641cef8-ce1d-4aaf-87b3-6f254350e27d" />

---

##  Tech Stack
- Python – Core language.
- Scikit-Learn – Data preprocessing and scaling.  
- XGBoost – Fraud detection classification model.
- Pandas & NumPy – Data handling and simulation.
- Seaborn & Matplotlib – Visualisations and insights. 
- Streamlit – Web app frontend & deployment.

### ML models I tried but did not use:
- Isolation Forests Model - This model failed to accurately detect Fraudulent transactions and classified Legit transactions as Fraud.
- Local Outlier Factor - Performance on this model was better but still nowhere near the level i hoped to ship the app with.

---
## Features
### Single Transaction Prediction

   - User enters transaction details manually (distance from home, online order, chip usage, etc.).
   - Model predicts fraud probability and displays a warning or success message.
   - Added XGBoost's Feature importance chart to show how it weighs different features
  
### Dataset Visualisations

- Fraud vs Normal distribution (class imbalance visualisation).
- Distance-based fraud behaviour:
     - Distance from home 
     - Distance from last transaction
- Ratio to median purchase price.
    - Categorical feature breakdowns:
	    - Repeat retailer
	    - Chip usage
	    - PIN usage
	    - Online order
- Each plot includes:
	- Value annotations
	- Text explanations

---

## ML Pipeline

1. Dataset Cleaning
    
    - Removed irrelevant/noisy features.
    - Handled categorical -> numeric conversions.
        
2. Feature Scaling
    
    - Standardised numeric features using **StandardScaler**.

3. Imbalanced Dataset Handling
    
    - Dataset was heavily **skewed** towards non-fraudulent cases.
    - Tackled using ML models that excel in Outlier detection

4. Model Choice
    
    - Evaluated multiple classifiers ( **Isolation Forest Classifier, Local Outlier Factor, XGBoost** ).
    - **XGBoost** chosen due to strong performance on imbalanced classification tasks.
        
5. Deployment
    
    - **Streamlit** app built with modular design (`Home.py` for prediction, `Visualisations.py` for insights).
    - Sidebar navigation between sections.
    - Optimised with caching (`@st.cache_data`, `@st.cache_resource`) to prevent reloading plots.

---

## Key Decisions & Challenges

- **Challenge:** Highly imbalanced dataset -> fraud cases were **<10%.**
    - **Decision:** Used XGBoost, added probability threshold analysis.
        
- **Challenge:** The initial dataset used [credit card fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) had **anonymized features (V1–V28)** for privacy.  
  - **Decision:** Switched to a **simulated dataset** with interpretable features (distance, chip usage, online order, etc.), making the app more **user-friendly and explainable**.

- **Challenge:** Visualisations reloading every time page switched.
    - **Decision:** **Cached** data & plots with Streamlit caching decorators.
        
- **Challenge:** Rendering large dataset in Streamlit was slow.
    - **Decision:** Worked with **50% random sample** for visuals, balancing speed and insight.
    
- **Challenge:** Making the app both **educational and practical**.
    - **Decision:** Added tooltips, expanders, and explanations next to every input & visualisation.

---

##  Future Iterations

- Try **SMOTE or ADASYN oversampling** to improve fraud detection recall.
- Extend/Replace dataset with **real-world transaction patterns**.
- Add an option to input bulk transactions for testing multiple transactions at once

---

##  Author

**Nishant Kumar**
Built during my internship at **Elevate Labs**  
GitHub: [r3nexe](https://github.com/R3Nexe)

---

##  Acknowledgements

- Dataset is simulated for academic purposes.
