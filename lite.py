import tensorflow as tf

# Load the existing model
model = tf.keras.models.load_model('deteksi_penyakit_ayam.h5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)