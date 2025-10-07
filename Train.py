import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# === Config ===
TICKER = "GGRM.JK"  # saham contoh
MODEL_PATH_KERAS = "backend/stock_model.keras"
MODEL_PATH_H5 = "backend/stock_model.h5"
SCALER_PATH = "backend/stock_scaler.pkl"

# === Download data ===
print(f"📥 Downloading data for {TICKER}...")
df = yf.download(TICKER, period="2y", interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# === Feature & Target ===
features = df[['Open', 'High', 'Low', 'Volume']].values
target = df['Close'].values

# === Scaling ===
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.reshape(-1, 1))

# === Sequence (windowing) ===
SEQ_LEN = 60
X, y = [], []
for i in range(SEQ_LEN, len(features_scaled)):
    X.append(features_scaled[i-SEQ_LEN:i])
    y.append(target_scaled[i])
X, y = np.array(X), np.array(y)

# === Train-test split ===
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === Build Model ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
print("🚀 Training model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# === Save model ===
os.makedirs("backend", exist_ok=True)
model.save(MODEL_PATH_KERAS)
model.save(MODEL_PATH_H5)
print(f"✅ Model disimpan ke {MODEL_PATH_KERAS} & {MODEL_PATH_H5}")

# === Save scaler ===
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler disimpan ke {SCALER_PATH}")
