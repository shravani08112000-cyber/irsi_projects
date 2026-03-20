from pathlib import Path
import joblib
import numpy as np
import streamlit as st

MODEL_PATH = Path("artifacts/models/model.pkl")

st.set_page_config(page_title="Project IRIS", page_icon="🌸", layout='centered')

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists:
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first"
        )
    return joblib.load(MODEL_PATH)

st.title("Project IRIS")
st.caption("A simple Streamlit app for IRIS species prediction")
st.write("Enet the flower measurements below and run a prediction")

with st.form("rediction_form"):

    sepal_length = st.number_input(
        "Sepal length (cm)", min_value=0.0, value=5.1, step=0.1
    )

    sepal_width = st.number_input(
        "Sepal width (cm)", min_value=0.0, value=3.5, step=0.1
    )

    petel_length = st.number_input(
        "petel length (cm)", min_value=0.0, value=1.4, step=0.1
    )

    petel_width = st.number_input(
        "petel width (cm)", min_value=0.0, value=0.2, step=0.1
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        model = load_model()
        input_data = np.array(
            [[sepal_length, sepal_width, petel_length, petel_width]], dtype=float
        )
        predction = model.predict(input_data)[0]
        st.success(f"Predcited Flower: {predction}")
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Predixtion Failed: {e}")