import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
import base64
import ServerService

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf

app = Flask(__name__)

service = ServerService.Service()

@app.route('/predict', methods=['POST'])
def predict_image():
    image_data = request.files['image'].read()
    predicted_char, score = service.predict_image(image_data=image_data)

    return jsonify({'predicted_char': predicted_char, 'score': float(score)})

if __name__ == '__main__':
    app.run(debug=True)
