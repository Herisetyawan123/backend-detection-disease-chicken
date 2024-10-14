import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Muat model TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Mendapatkan input dan output detail dari interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kategorisasi
class_labels = ['kolera', 'sehat', 'snot']  # Sesuaikan dengan label model

def load_and_preprocess_image(frame):
    """Preprocess gambar dari frame video OpenCV."""
    # Resize frame menjadi ukuran yang sesuai dengan input model (misal: 150x150)
    img = Image.fromarray(frame).resize((150, 150))
    # Convert to RGB jika gambar memiliki 4 channel (RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalisasi gambar (sesuai dengan yang dilakukan ImageDataGenerator)
    img_array = img_array / 255.0
    return img_array

def classify_frame(frame):
    """Klasifikasikan frame video dan kembalikan hasilnya."""
    img_array = load_and_preprocess_image(frame)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Jalankan inferensi
    interpreter.invoke()

    # Dapatkan hasil prediksi dari output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100  # Confidence score
    return class_labels[predicted_class], confidence

def generate_frames():
    camera = cv2.VideoCapture(0)  # Use the camera
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            # Lakukan klasifikasi pada frame
            category, confidence = classify_frame(frame)

            if confidence > 90:
                # Tampilkan hasil deteksi di atas frame (tuliskan di video)
                label = f"{category} ({confidence:.2f}%)"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Use a generator to yield the frame to the Flask response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
