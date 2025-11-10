# ekpc_load_forecast_app.py

# ===========================
# 1. Import Libraries
# ===========================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import numpy as np

# ===========================
# 2. Load Data
# ===========================
df = pd.read_csv("ekpc_usage.csv")  # Replace with your CSV file path
df['Datetime'] = pd.to_datetime(df['Datetime'])

# ===========================
# 3. Feature Extraction
# ===========================
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month

# Lag features
df['Lag1'] = df['EKPC_MW'].shift(1)
df['Lag2'] = df['EKPC_MW'].shift(2)
df['Lag3'] = df['EKPC_MW'].shift(3)

# Drop rows with NaN
df_model = df.dropna().reset_index(drop=True)

# Features and target
X = df_model[['Lag3', 'Lag2', 'Lag1']]
y = df_model['EKPC_MW']

# ===========================
# 4. Train/Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ===========================
# 5. Train Random Forest Model
# ===========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===========================
# 6. Evaluate Model
# ===========================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# ===========================
# 7. Plot Predictions vs Actual
# ===========================
plt.figure(figsize=(15,5))
plt.plot(y_test.values[:200], label='Actual', color='blue')
plt.plot(y_pred[:200], label='Predicted', color='red')
plt.title("EKPC Load Forecasting: Actual vs Predicted (First 200 Hours)")
plt.xlabel("Hour")
plt.ylabel("Usage (MW)")
plt.legend()
plt.show()

# ===========================
# 8. Streamlit Web App
# ===========================
st.title("AI-Based Load Forecasting")

st.write("Enter the usage for the last 3 hours to predict the next hour load:")

lag3 = st.number_input("Usage 3 hours ago (MW)", value=float(df_model['Lag3'].iloc[-1]))
lag2 = st.number_input("Usage 2 hours ago (MW)", value=float(df_model['Lag2'].iloc[-1]))
lag1 = st.number_input("Usage 1 hour ago (MW)", value=float(df_model['Lag1'].iloc[-1]))

if st.button("Predict Next Hour Usage"):
    input_data = np.array([[lag3, lag2, lag1]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted EKPC Usage for Next Hour: {prediction:.2f} MW")

st.write("---")
st.write("Model Performance on Test Set:")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

st.write("Visualization of Test Predictions:")
st.line_chart(pd.DataFrame({
    'Actual': y_test.values[:200],
    'Predicted': y_pred[:200]
}))
