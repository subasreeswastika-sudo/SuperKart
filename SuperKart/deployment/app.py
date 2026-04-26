import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# 1. Page Configuration & Title
st.set_page_config(page_title="SuperKart Sales Predictor", layout="centered")
st.title("📈 SuperKart Weekly Sales Forecast")
st.markdown("""
This application uses an automated MLOps pipeline to predict total revenue 
based on product attributes and store characteristics.
""")

# 2. Model Loading from Hugging Face Hub
@st.cache_resource # Cache the model so it doesn't reload on every button click
def load_model():
    # Replace with your actual HF username/repo
    REPO_ID = "swastisubi/SuperKart" 
    FILENAME = "model.joblib"
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="model")
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. User Input Form
st.sidebar.header("Input Product & Store Details")

def get_user_input():
    # Product Details
    p_weight = st.sidebar.number_input("Product Weight", min_value=0.0, value=12.0)
    p_sugar = st.sidebar.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    p_area = st.sidebar.slider("Allocated Display Area", 0.0, 0.2, 0.05)
    p_type = st.sidebar.selectbox("Product Category", 
                                  ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 
                                   'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 
                                   'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 
                                   'Breads', 'Starchy Foods', 'Others', 'Seafood'])
    p_mrp = st.sidebar.number_input("Maximum Retail Price (MRP)", min_value=0.0, value=150.0)
    
    # Store Details
    s_year = st.sidebar.number_input("Establishment Year", min_value=1980, max_value=2026, value=2010)
    s_size = st.sidebar.selectbox("Store Size", ["High", "Medium", "Small"])
    s_city = st.sidebar.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    s_type = st.sidebar.selectbox("Store Type", 
                                  ['Supermarket Type1', 'Supermarket Type2', 
                                   'Food Mart', 'Departmental Store'])

    # Map inputs to DataFrame (Must match the training feature order)
    data = {
        'Product_Weight': p_weight,
        'Product_Sugar_Content': p_sugar,
        'Product_Allocated_Area': p_area,
        'Product_Type': p_type,
        'Product_MRP': p_mrp,
        'Store_Establishment_Year': s_year,
        'Store_Size': s_size,
        'Store_Location_City_Type': s_city,
        'Store_Type': s_type
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# 4. Inference & Output Evaluation
st.subheader("Current Input Data")
st.write(input_df)

if st.button("Predict Total Sales"):
    prediction = model.predict(input_df)
    
    # Visualizing the output
    st.metric(label="Predicted Total Revenue", value=f"₹{prediction[0]:,.2f}")
    
    if prediction[0] > 4000:
        st.info("💡 High Volume Alert: This product/store combination is projected to perform above average.")
    else:
        st.warning("📉 Low Volume Alert: Consider optimizing display area or pricing strategies.")

# 5. Footer & Links
st.markdown("---")
st.caption("Developed by Subasree | Part of the SuperKart MLOps Project")
