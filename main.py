from fastapi import FastAPI
import yfinance as yf

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Stock Prediction API is running 🚀"}

@app.get("/predict")
def predict(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")

        if data.empty:
            return {"error": "Data not available for symbol"}

        # ambil harga penutupan terakhir
        last_price = data["Close"].iloc[-1]

        return {"symbol": symbol, "price": float(last_price)}

    except Exception as e:
        return {"error": str(e)}
