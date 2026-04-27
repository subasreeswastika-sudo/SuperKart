import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title="SuperKart Sales Predictor", layout="centered", page_icon="📈")

st.title("📈 SuperKart Weekly Sales Forecast")
st.markdown("This application predicts **total revenue** using an XGBoost model.")

# ==========================
# Model Loading with Better Handling
# ==========================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    REPO_ID = "swastisubi/SuperKart"
    FILENAME = "model.joblib"
    
    token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
    
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="model",
            token=token
        )
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully from Hugging Face!")
        return model
    except Exception as e:
        error_str = str(e).lower()
        if "401" in error_str or "gated" in error_str or "restricted" in error_str:
            st.error("❌ Cannot access the model (401 Unauthorized)")
            st.info("**Solution:**\n"
                    "1. Go to your model repo → Settings → Set visibility to **Public**\n"
                    "OR\n"
                    "2. Add `HF_TOKEN` secret in your Space Settings → Variables and secrets")
        else:
            st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ==========================
# Rest of your app (User Input, Prediction, Feature Importance)
# ==========================
st.sidebar.header("Input Product & Store Details")

def get_user_input():
    p_weight = st.sidebar.number_input("Product Weight (kg)", min_value=0.0, value=12.0, step=0.1)
    p_sugar = st.sidebar.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    p_area = st.sidebar.slider("Allocated Display Area (%)", 0.0, 0.2, 0.05, step=0.001)
    p_type = st.sidebar.selectbox("Product Category", [
        'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
        'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
        'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
        'Starchy Foods', 'Others', 'Seafood'
    ])
    p_mrp = st.sidebar.number_input("Maximum Retail Price (MRP)", min_value=0.0, value=150.0, step=1.0)

    s_year = st.sidebar.number_input("Store Establishment Year", min_value=1980, max_value=2026, value=2010)
    s_size = st.sidebar.selectbox("Store Size", ["High", "Medium", "Small"])
    s_city = st.sidebar.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    s_type = st.sidebar.selectbox("Store Type", ['Supermarket Type1', 'Supermarket Type2', 'Food Mart', 'Departmental Store'])
    s_id = st.sidebar.selectbox("Store ID", ["OUT049", "OUT018", "OUT046", "OUT035", "OUT045", "OUT027"])

    data = {
        'Product_Weight': p_weight,
        'Product_Sugar_Content': p_sugar,
        'Product_Allocated_Area': p_area,
        'Product_Type': p_type,
        'Product_MRP': p_mrp,
        'Store_Establishment_Year': s_year,
        'Store_Size': s_size,
        'Store_Location_City_Type': s_city,
        'Store_Type': s_type,
        'Store_Id': s_id
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

tab1, tab2 = st.tabs(["📋 Input & Prediction", "📊 Feature Importance"])

with tab1:
    st.subheader("Current Input Data")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Predict Total Sales", type="primary"):
        with st.spinner("Predicting..."):
            prediction = model.predict(input_df)
            pred_value = float(prediction[0])
            
            st.success("✅ Prediction Complete!")
            st.metric(label="Predicted Total Revenue", value=f"₹{pred_value:,.2f}")
            
            if pred_value > 4000:
                st.info("💡 High Volume Alert: Strong performer expected!")
            else:
                st.warning("📉 Low Volume Alert: Optimization may be needed.")

            # Download option
            csv = input_df.copy()
            csv["Predicted_Revenue"] = pred_value
            st.download_button(
                label="📥 Download Prediction as CSV",
                data=csv.to_csv(index=False).encode('utf-8'),
                file_name="superkart_prediction.csv",
                mime="text/csv"
            )

with tab2:
    st.subheader("Feature Importance")
    if st.button("Show Top 15 Features"):
        try:
            preprocessor = model.named_steps['preprocessor']
            xgb_model = model.named_steps['xgb']
            feature_names = preprocessor.get_feature_names_out()
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top 15 Features")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp_df, use_container_width=True)
        except Exception as e:
            st.error(f"Feature importance unavailable: {str(e)}")

st.markdown("---")
st.caption("Developed by Subasree | SuperKart MLOps Project")
