import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="swastisubi/SuperKart", filename="model.joblib")
model = joblib.load(model_path)

# Streamlit UI for SuperKart Sales Prediction
st.title("SuperKart Sales Prediction App")
st.write("""
This application predicts sales based on product and store characteristics.
Please enter the product and store data below to get a prediction.
""")

# User input
st.header("Product Information")
product_weight = st.number_input("Product Weight (kg)", min_value=0.1, max_value=30.0, value=10.0, step=0.1)
product_sugar_content = st.selectbox("Product Sugar Content", ["low sugar", "regular", "no sugar"])
product_allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
product_type = st.selectbox("Product Type", [
    "meat", "snack foods", "hard drinks", "dairy", "canned", "soft drinks",
    "health and hygiene", "baking goods", "bread", "breakfast", "frozen foods",
    "fruits and vegetables", "household", "seafood", "starchy foods", "others"
])
product_mrp = st.number_input("Product MRP (Maximum Retail Price)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)

st.header("Store Information")
store_id = st.selectbox("Store ID", ["Store_01", "Store_02", "Store_03", "Store_04", "Store_05"])
store_establishment_year = st.number_input("Store Establishment Year", min_value=1950, max_value=2023, value=2000, step=1)
store_size = st.selectbox("Store Size", ["high", "medium", "low"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_Type': product_type,
    'Product_MRP': product_mrp,
    'Store_Id': store_id,
    'Store_Establishment_Year': store_establishment_year,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type
}])


if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    # The model is now a regressor, so we directly display the predicted sales value.
    st.subheader("Prediction Result:")
    st.success(f"Predicted Sales: **${prediction:.2f}**")
