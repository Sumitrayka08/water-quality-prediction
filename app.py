import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("water_quality_model.pkl")

st.title("💧 Water Quality Prediction")

st.write("Enter water parameters:")

ph = st.number_input("pH", 0.0, 14.0)
hardness = st.number_input("Hardness")
solids = st.number_input("Solids")
chloramines = st.number_input("Chloramines")
sulfate = st.number_input("Sulfate")
conductivity = st.number_input("Conductivity")
organic_carbon = st.number_input("Organic Carbon")
trihalomethanes = st.number_input("Trihalomethanes")
turbidity = st.number_input("Turbidity")

if st.button("Predict"):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                            conductivity, organic_carbon,
                            trihalomethanes, turbidity]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Water is SAFE to drink")
    else:
        st.error("❌ Water is NOT safe to drink")