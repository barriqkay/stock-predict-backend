import tensorflow as tf

# Load model lama
model = tf.keras.models.load_model('stock_model_old.h5', compile=False)

# Simpan ulang dengan format H5 baru
model.save('stock_model.h5', save_format='h5')
