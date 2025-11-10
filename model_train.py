# train_lstm.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load data
df = pd.read_csv("ekpc_usage.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime')

# 2. Create lag features (t-3, t-2, t-1)
df['Lag1'] = df['EKPC_MW'].shift(1)
df['Lag2'] = df['EKPC_MW'].shift(2)
df['Lag3'] = df['EKPC_MW'].shift(3)
df = df.dropna().reset_index(drop=True)

# 3. Prepare data for LSTM
X = df[['Lag3','Lag2','Lag1']].values
y = df['EKPC_MW'].values

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Train/test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1)
])

# Compile with explicit optimizer configuration
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# 6. Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# 7. Evaluate model
y_pred = model.predict(X_test)
mae = np.mean(np.abs(y_test - y_pred.flatten()))
r2 = 1 - np.sum((y_test - y_pred.flatten())**2)/np.sum((y_test - np.mean(y_test))**2)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Save trained model
model.save("ekpc_lstm_model.h5", save_format='h5')
print("Model saved as ekpc_lstm_model.h5")
