import tensorflow as tf
MODEL_PATH = "stock_model.h5"

# Load model H5
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
