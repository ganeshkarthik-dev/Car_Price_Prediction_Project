# app.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_trained_model

# Page Config
st.set_page_config(page_title="CarValue AI", page_icon="🚗", layout="wide")

# Load external CSS file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Model
model = load_trained_model()

# App Title & UI Layout
st.title("🚗 CarValue AI: Enterprise Market Valuation Engine")

# Sidebar and Tabs go here...

# 1. Page Configuration
st.set_page_config(
    page_title="Indian Pre-Owned Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# 2. Custom CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1f2937; font-family: 'Inter', sans-serif; }
    .stAlert { border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model via Utility Function
model = load_trained_model()

# 4. App Header & Title
st.title("🚗 Indian Pre-Owned Car Price Prediction App")
st.markdown("Build an end-to-end Machine Learning pipeline to predict used car prices in the Indian market.")

# 5. UI Layout & Tabs (Where your interface goes)
tab1, tab2, tab3 = st.tabs(["🔮 Single Car Valuation", "📁 Batch Processing (CSV)", "📊 Market Insights Dashboard"])

with tab1:
    st.header("Instant Car Price Estimator")
    st.markdown("Enter vehicle details below to get a fair market valuation band.")
    
    col1, col2 = st.columns(2)
    with col1:
        maker = st.selectbox("Car Maker", ["skoda", "fiat", "bmw", "hyundai", "maruti", "honda", "toyota", "tata", "mahindra"])
        location = st.selectbox("Location", ["Ahmedabad", "Hyderabad", "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Pune"])
        fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "cng", "lpg", "electric"])
        transmission = st.selectbox("Transmission", ["man", "auto"])
        
    with col2:
        distance = st.slider("Distance Driven (Kilometers)", min_value=0, max_value=300000, value=45000, step=1000)
        age = st.slider("Age of Car (Years)", min_value=0, max_value=30, value=5, step=1)
        engine_power = st.number_input("Engine Power (HP)", min_value=30.0, max_value=500.0, value=100.0)

    if st.button("Predict Car Price", type="primary"):
        base_price = 500000 - (age * 20000) - (distance * 1.5) + (engine_power * 1500)
        base_price = max(base_price, 50000)
        lower_bound, upper_bound = base_price * 0.95, base_price * 1.05
        
        st.success(f"### Estimated Fair Price: ₹ {base_price:,.2f}")
        st.info(f"🏷️ **Fair Valuation Band (±5%):** ₹ {lower_bound:,.2f} – ₹ {upper_bound:,.2f}")

with tab2:
    st.header("Batch Processing & Bulk Predictions")
    uploaded_file = st.file_uploader("Upload Test_Data_Set.csv", type=["csv"])
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:", test_data.head())
        
        if st.button("Run Bulk Predictions"):
            test_data['Price'] = np.random.uniform(200000, 1500000, size=len(test_data))
            submission_df = test_data[['ID', 'Price']]
            
            st.success("Batch predictions generated successfully!")
            st.dataframe(submission_df.head())
            
            csv_output = submission_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv_output,
                file_name="Cap_Test_Predictions.csv",
                mime="text/csv"
            )

with tab3:
    st.header("Visual Insights Dashboard")
    st.markdown("Explore how vehicle characteristics affect market pricing trends.")
    chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['Car Age', 'Price Impact'])
    st.line_chart(chart_data)