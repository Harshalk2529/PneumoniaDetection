from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import joblib
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

labels = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 150

def prepare(image_data):
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.convert('L')
    img_array = np.array(img)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    normalized_array = new_array / 255.0
    return normalized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

cnn_model = tf.keras.models.load_model("/PneumoniaDetection/cnn_model.keras")
symp_model = joblib.load('/PneumoniaDetection/pneumonia_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    data = request.get_json()
    image_data = data['image']
    img = prepare(image_data)
    prediction = cnn_model.predict(img)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0
    result = labels[predicted_class]
    return jsonify({'prediction': result})

@app.route('/predict_hybrid', methods=['POST'])
def predict_hybrid():
    data = request.get_json()
    image_data = data['image']
    symptoms = data['symptoms']
    img = prepare(image_data)
    cnn_prediction = cnn_model.predict(img)
    cnn_class = 1 if cnn_prediction[0][0] > 0.5 else 0
    symp_prediction = symp_model.predict(np.array([symptoms]))

    hybrid_prediction = 0.1 * symp_prediction[0][0] + 0.9 * cnn_prediction[0][0]
    predicted_class = 1 if hybrid_prediction > 0.5 else 0
    result = labels[predicted_class]
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)