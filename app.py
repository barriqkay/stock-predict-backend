from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Path ke model H5 (pastikan sudah push ke repo)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stock_model.h5")

# Load model tanpa compile untuk menghindari error
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model berhasil dimuat!")
except Exception as e:
    print("Gagal load model:", e)
    raise e

@app.route("/")
def home():
    return "API Stock Prediction Online!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON contoh:
    {
        "data": [[0.1, 0.2, 0.3, 0.4], ...]  # shape (30, 4)
    }
    """
    try:
        data = request.json["data"]
        data = np.array(data, dtype=np.float32)

        # Pastikan shape sesuai (1, 30, 4)
        if data.shape != (30, 4):
            return jsonify({"error": "Data harus memiliki shape (30, 4)"}), 400
        data = np.expand_dims(data, axis=0)

        # Prediksi
        prediction = model.predict(data)
        prediction_value = float(prediction[0][0])

        return jsonify({"prediction": prediction_value})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
