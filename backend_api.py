from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import yfinance as yf
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock_api")

app = FastAPI()

# Path model & scaler (karena file ada di folder backend/)
MODEL_PATH = "stock_model.keras"
SCALER_PATH = "scaler_x.pkl"

# Load model & scaler
logger.info("Loading model and scaler...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler_x = joblib.load(SCALER_PATH)
logger.info("Model and scaler loaded successfully.")

# Schema input
class StockInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float

@app.post("/predict")
def predict_stock(data: StockInput):
    features = np.array([[data.open, data.high, data.low, data.volume]])
    features_scaled = scaler_x.transform(features)
    prediction = model.predict(features_scaled)
    return {"predicted_close": float(prediction[0][0])}

@app.get("/history/{ticker}")
def get_stock_history(ticker: str, period: str = "1mo", interval: str = "1d"):
    """
    Ambil histori harga saham dari Yahoo Finance
    contoh: /history/GGRM.JK?period=1mo&interval=1d
    """
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df = df.reset_index()
        history = []
        for _, row in df.iterrows():
            history.append({
                "date": str(row["Date"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            })
        return {"ticker": ticker, "history": history}
    except Exception as e:
        return {"error": str(e)}
