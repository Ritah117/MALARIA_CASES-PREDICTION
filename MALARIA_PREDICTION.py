import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your malaria prediction model (make sure the model is saved in the same directory)
model = joblib.load('malaria_model.joblib')

# Title of the application
st.title("Malaria Prediction App")

# Tabs for different sections
tabs = st.tabs(["Description and Creators", "Predict"])

# Description and Creators Tab
with tabs[0]:
    st.header("Description")
    st.write("""
    This application predicts positive malaria cases using historical data.
    The model is built using a Random Forest algorithm, trained on various facilities' malaria case data over several years.
    The predictions can help in planning and resource allocation in healthcare.
    """)

    st.header("Creators")
    st.write("""
    - **ANNRITA MUKAMI**
    - **MARQULINE OPIYO**
    """)

# Predict Tab
with tabs[1]:
    st.header("Predict Malaria Cases")

    # Input fields for user
    year = st.number_input("Year (e.g., 2024)", min_value=2024, max_value=2034, value=2024)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1)

    # Lag features (assuming they are always 0 initially for new predictions)
    lag_1 = st.number_input("Lag 1 (previous cases)", min_value=0, value=0)
    lag_2 = st.number_input("Lag 2 (two months ago)", min_value=0, value=0)
    lag_3 = st.number_input("Lag 3 (three months ago)", min_value=0, value=0)

    # Button to make prediction
    if st.button("Predict"):
        # Prepare the input data
        input_data = np.array([[year, month, lag_1, lag_2, lag_3]])

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction result
        st.success(f"Predicted Positive Malaria Cases: {int(prediction[0])}")

# To run the app, use the command in terminal: streamlit run your_script_name.py
