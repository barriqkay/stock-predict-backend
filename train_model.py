import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Konfigurasi
TICKER = "GGRM.JK"   # saham Gudang Garam
PERIOD = "5y"
SEQ_LEN = 60
HORIZON = 1

def fetch_and_prepare(ticker=TICKER, period=PERIOD):
    print("Fetching data and preparing sequences...")
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Periksa ticker atau koneksi internet.")

    df = df[['Open','High','Low','Close','Volume']].dropna()

    # Tambah fitur teknikal
    df['return1'] = df['Close'].pct_change(1)
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['std7'] = df['Close'].rolling(7).std()
    df = df.dropna()

    features = ['Close','Open','High','Low','Volume','return1','ma7','ma21','std7']
    data = df[features].values.astype(float)
    targets = df['Close'].shift(-HORIZON).values.astype(float)

    # Hapus baris dengan target NaN pakai index
    idx = np.where(~np.isnan(targets))[0]
    data = data[idx, :]
    targets = targets[idx]

    print("DEBUG shapes → data:", data.shape, "targets:", targets.shape)

    X, y = [], []
    for i in range(SEQ_LEN, len(data)):
        X.append(data[i-SEQ_LEN:i])
        y.append(targets[i])
    return np.array(X), np.array(y), df

# Ambil data
X, y, df_full = fetch_and_prepare()
print("Data siap untuk training:", X.shape, y.shape)

# Bangun model LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
print(model.summary())

# Training model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Simpan model dengan format baru
model.save("stock_model.keras", save_format="keras")

