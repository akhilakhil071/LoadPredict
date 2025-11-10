# ekpc_lstm_app.py

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# 1. Load trained LSTM model
model = load_model("ekpc_lstm_model.h5")

st.title("EKPC Load Forecasting (LSTM)")

st.write("Enter usage for the last 3 hours to predict next-hour load:")

# 2. Input fields
lag3 = st.number_input("Usage 3 hours ago (MW)")
lag2 = st.number_input("Usage 2 hours ago (MW)")
lag1 = st.number_input("Usage 1 hour ago (MW)")

# 3. Predict next hour usage
if st.button("Predict Next Hour Usage"):
    x_input = np.array([[lag3, lag2, lag1]]).reshape((1,3,1))
    pred = model.predict(x_input)[0][0]
    st.success(f"Predicted EKPC Usage: {pred:.2f} MW")
