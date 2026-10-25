# 🚗 Indian Pre-Owned Car Price Prediction App

An end-to-end Machine Learning web application built with Streamlit and XGBoost/Random Forest to predict pre-owned car prices in the Indian market. Developed as part of the Unlox Machine Learning Capstone.

## 🚀 Features
* **Instant Car Valuation:** Dynamic price estimation based on vehicle brand, location, age, distance driven, and engine power.
* **Batch Processing:** Upload a test CSV dataset (`Cap_Test_Data_2025.csv`) to execute bulk predictions and download certified results.
* **Market Insights Dashboard:** Interactive visual analytics exploring vehicle depreciation trends and pricing distributions.

## 🛠️ Project Structure
```text
Car_Price_Prediction_Project/
├── data/
│   ├── Cap_Training_Data_2025.csv
│   ├── Cap_Test_Data_2025.csv
│   └── Cap_Sample_Submission_2025.csv
├── app.py                 # Streamlit UI & Layout
├── utils.py               # Backend model loading & utilities
├── style.css              # Custom enterprise styling
├── car_price_model.pkl    # Trained Machine Learning Model
└── requirements.txt       # Project dependencies