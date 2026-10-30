
import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer details to predict the probability "
    "that the customer will leave the bank."
)

# --------------------------------------------------
# File paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_bank_churn_model.keras"
PREPROCESSOR_PATH = BASE_DIR / "bank_churn_preprocessor.pkl"


# --------------------------------------------------
# Load trained model and fitted preprocessor
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: {PREPROCESSOR_PATH.name}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # This requires a compatible scikit-learn version
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error("Unable to load the model or preprocessor.")
    st.exception(e)
    st.info(
        "If the error mentions _RemainderColsList, recreate "
        "the preprocessor using a compatible scikit-learn "
        "environment and save it again."
    )
    st.stop()


# --------------------------------------------------
# Customer input form
# --------------------------------------------------
with st.form("customer_form"):

    country = st.selectbox(
        "Country",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0
    )

    products_number = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    credit_card = st.selectbox(
        "Has Credit Card?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    active_member = st.selectbox(
        "Is Active Member?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

    submitted = st.form_submit_button("Predict Churn")


# --------------------------------------------------
# Make prediction
# --------------------------------------------------
if submitted:

    # Column names and order must match training data
    input_data = pd.DataFrame([{
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary
    }])

    try:
        input_transformed = preprocessor.transform(input_data)

        probability = float(
            model.predict(input_transformed, verbose=0).ravel()[0]
        )

        st.subheader("Prediction Result")
        st.metric(
            "Estimated Churn Probability",
            f"{probability:.2%}"
        )

        if probability >= 0.5:
            st.error("Prediction: Customer is likely to leave.")
        else:
            st.success("Prediction: Customer is likely to stay.")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

    st.caption(
        "The result is a model estimate, not a guarantee "
        "of customer behavior."
    )
