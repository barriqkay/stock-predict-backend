import tensorflow as tf

# Load model lama (format .keras)
model = tf.keras.models.load_model("stock_model.keras")

# Simpan ulang menjadi format H5
model.save("stock_model.h5")
print("Model berhasil disimpan sebagai stock_model.h5")
