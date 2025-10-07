import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Konfigurasi default
DEFAULT_TICKER = "GGRM.JK"
PERIOD = "5y"
SEQ_LEN = 60

def load_data(ticker=DEFAULT_TICKER, period=PERIOD, seq_len=SEQ_LEN):
    print(f"\nFetching data for {ticker} ...")
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        raise RuntimeError(f"⚠️ Tidak ada data untuk {ticker}. Periksa kode saham atau koneksi internet.")

    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, df

def evaluate_model(ticker):
    # Load data
    X, y, scaler, df = load_data(ticker)

    # Load model
    print("\n📥 Loading trained model...")
    model = tf.keras.models.load_model("stock_model.h5")

    # Prediksi
    y_pred = model.predict(X, verbose=0)

    # Balik ke harga asli
    y_inv = scaler.inverse_transform(y.reshape(-1,1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Hitung metrik evaluasi
    mae = mean_absolute_error(y_inv, y_pred_inv)
    rmse = mean_squared_error(y_inv, y_pred_inv, squared=False)
    mape = np.mean(np.abs((y_inv - y_pred_inv) / y_inv)) * 100

    print("\n📊 Evaluation Metrics:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # Plot hasil
    plt.figure(figsize=(12,6))
    plt.plot(y_inv, label="Actual Price")
    plt.plot(y_pred_inv, label="Predicted Price")
    plt.title(f"{ticker} Stock Price: Actual vs Predicted")
    plt.xlabel("Time Steps")
    plt.ylabel("Price (IDR)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("=== Stock Price Evaluation ===")
    ticker = input(f"Masukkan kode saham (default {DEFAULT_TICKER}): ").strip()
    if ticker == "":
        ticker = DEFAULT_TICKER
    evaluate_model(ticker)
