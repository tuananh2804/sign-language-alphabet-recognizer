import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
import base64
import ServerService

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
import random

app = Flask(__name__)

service = ServerService.Service()

@app.route('/predict', methods=['POST'])
def predict_image():
    image_data = request.files['image'].read() 
    
    predicted_char, score = service.predict_image(image_data=image_data)
    
    print(predicted_char, end="\n");

    return jsonify({'predicted_char': predicted_char, 'score': float(score)})
    
    # random_number = 1
    # if random_number == 1:
    #     print("1", end="\n")
    #     return jsonify({'predicted_char': 'b', 'score': 'float(0.5)'})
    # elif random_number == 2:
    #     print("2", end="\n")
    #     return jsonify({'predicted_char': 'space', 'score': 'float(0.5)'})
    # else:
    #     print("3", end="\n")
    #     return jsonify({'predicted_char': 'del', 'score': 'float(0.5)'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
