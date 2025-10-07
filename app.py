from flask import Flask, request, jsonify
import tensorflow as tf
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import InputLayer

SEQ_LEN = 60
MODEL_PATH = "stock_model.keras"

# Custom function untuk menghapus 'batch_shape' saat load model lama
def custom_input_layer(*args, **kwargs):
    kwargs.pop('batch_shape', None)
    return InputLayer(*args, **kwargs)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'InputLayer': custom_input_layer})
model.make_predict_function()

app = Flask(__name__)

def prepare_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        raise ValueError(f"Tidak ada data untuk {ticker}")
    df = df[['Close']].dropna()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    last_sequence = data_scaled[-SEQ_LEN:]
    X = np.array(last_sequence).reshape(1, SEQ_LEN, 1)
    return X, scaler, df

@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("ticker", default="GGRM.JK")
    try:
        X, scaler, df = prepare_data(ticker)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return jsonify({
            "ticker": ticker,
            "last_close": float(df['Close'].iloc[-1]),
            "predicted_next": float(pred_price)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
