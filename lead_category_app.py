
import streamlit as st
import pickle
import pandas as pd

# Load the trained model and label encoders
model = pickle.load(open("lead_category_model.pkl", "rb"))
le_product = pickle.load(open("product_id_encoder.pkl", "rb"))
le_source = pickle.load(open("source_encoder.pkl", "rb"))
le_sales_agent = pickle.load(open("sales_agent_encoder.pkl", "rb"))
le_location = pickle.load(open("location_encoder.pkl", "rb"))
le_category = pickle.load(open("label_encoder.pkl", "rb"))

# Streamlit App UI
st.title("🔍 Lead Category Prediction App")
st.sidebar.header("Input Lead Details")

product_id = st.sidebar.selectbox("Product ID", le_product.classes_)
source = st.sidebar.selectbox("Source", le_source.classes_)
sales_agent = st.sidebar.selectbox("Sales Agent", le_sales_agent.classes_)
location = st.sidebar.selectbox("Location", le_location.classes_)
delivery_mode = st.sidebar.selectbox("Delivery Mode", ["Mode-1", "Mode-2", "Mode-3", "Mode-4", "Mode-5"])
created_year = st.sidebar.number_input("Created Year", min_value=2000, max_value=2030, value=2024)
created_month = st.sidebar.number_input("Created Month", min_value=1, max_value=12, value=5)
created_day = st.sidebar.number_input("Created Day", min_value=1, max_value=31, value=10)
created_hour = st.sidebar.number_input("Created Hour", min_value=0, max_value=23, value=12)

# Prepare input data
input_data = pd.DataFrame([{
    "Product_ID": le_product.transform([product_id])[0],
    "Source": le_source.transform([source])[0],
    "Sales_Agent": le_sales_agent.transform([sales_agent])[0],
    "Location": le_location.transform([location])[0],
    "Created_Year": created_year,
    "Created_Month": created_month,
    "Created_Day": created_day,
    "Created_Hour": created_hour,
    "Delivery_Mode_Mode-2": 1 if delivery_mode == "Mode-2" else 0,
    "Delivery_Mode_Mode-3": 1 if delivery_mode == "Mode-3" else 0,
    "Delivery_Mode_Mode-4": 1 if delivery_mode == "Mode-4" else 0,
    "Delivery_Mode_Mode-5": 1 if delivery_mode == "Mode-5" else 0
}])

# Add missing columns if any
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure column order
input_data = input_data[model.feature_names_in_]

# Predict
if st.button("Predict Lead Category"):
    prediction = model.predict(input_data)[0]
    category = le_category.inverse_transform([prediction])[0]
    st.success(f"🧠 Predicted Lead Category: **{category}**")
