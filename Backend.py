import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# ======================
# Path penyimpanan
# ======================
MODEL_KERAS_PATH = os.path.join("backend", "stock_model.keras")
MODEL_H5_PATH = os.path.join("backend", "stock_model.h5")
SCALER_X_PATH = os.path.join("backend", "scaler_X.pkl")
SCALER_Y_PATH = os.path.join("backend", "scaler_Y.pkl")

# ======================
# Ambil data historis
# ======================
ticker = "GGRM.JK"
period = "3y"
df = yf.download(ticker, period=period, interval="1d")
df = df[['Open', 'High', 'Low', 'Volume', 'Close']].dropna()

# ======================
# Scaling fitur & target
# ======================
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = df[['Open', 'High', 'Low', 'Volume']].values
y = df[['Close']].values

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scaler
joblib.dump(scaler_x, SCALER_X_PATH)
joblib.dump(scaler_y, SCALER_Y_PATH)

# ======================
# Buat sequence (LSTM input)
# ======================
SEQ_LEN = 30  # panjang sequence
X_seq, y_seq = [], []

for i in range(len(X_scaled) - SEQ_LEN):
    X_seq.append(X_scaled[i:i+SEQ_LEN])
    y_seq.append(y_scaled[i+SEQ_LEN])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)

# ======================
# Bangun Model LSTM
# ======================
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ======================
# Training
# ======================
history = model.fit(
    X_seq, y_seq,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ======================
# Save model
# ======================
model.save(MODEL_KERAS_PATH)
model.save(MODEL_H5_PATH)

print(f"✅ Model saved to {MODEL_KERAS_PATH} and {MODEL_H5_PATH}")
print(f"✅ Scalers saved to {SCALER_X_PATH} and {SCALER_Y_PATH}")
