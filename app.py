
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ORP House Rent Predictor",
    page_icon="🏠",
    layout="centered"
)

# App heading
st.title("🏠ORP House Rent Prediction")
st.write("Predict the estimated monthly rent of a property using Machine Learning.")

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
MODEL_PATH = BASE_DIR / "rent_model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please train and save your model first.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# Input form
st.subheader("Enter Property Details")

with st.form("rent_prediction_form"):
    location = st.text_input("Location", placeholder="e.g. Mumbai")
    size = st.number_input(
        "Size (in square feet)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=50
    )
    bhk = st.selectbox("Number of BHK", [1, 2, 3, 4, 5])
    bathroom = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
    furnishing = st.selectbox(
        "Furnishing Status",
        ["Unfurnished", "Semi-Furnished", "Furnished"]
    )
    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "Independent House", "Villa"]
    )
    tenant = st.selectbox(
        "Preferred Tenant",
        ["Bachelors", "Bachelors/Family", "Family"]
    )
    submit = st.form_submit_button("Predict Monthly Rent")

# Prediction
if submit:
    if not location.strip():
        st.warning("Please enter the property location.")
    else:
        try:
            input_data = pd.DataFrame([{
                "BHK": bhk,
                "Size": size,
                "Bathroom": bathroom,
                "Floor": "Ground out of 1",
                "Area Type": "Super Area",
                "Area Locality": location.strip(),
                "City": location.strip(),
                "Furnishing Status": furnishing,
                "Tenant Preferred": tenant,
                "Point of Contact": "Contact Owner"
            }])

            # Add property type if the trained model uses this feature
            # Your model must receive the same features used during training.
            prediction = model.predict(input_data)[0]

            st.success("Rent prediction completed!")
            st.metric(
                label="Estimated Monthly Rent",
                value=f"₹{max(0, prediction):,.0f}"
            )
            st.caption(
                "This is an estimate. Actual rent may vary by locality "
                "and property conditions."
            )

        except Exception as e:
            st.error(
                "Prediction failed because the input features may not "
                "match those used to train your model."
            )
            st.code(str(e))
