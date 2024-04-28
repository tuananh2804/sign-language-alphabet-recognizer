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
    
    
    # # Lưu ảnh vào thư mục "image_request"
    # image_folder = 'image_request'
    # if not os.path.exists(image_folder):
    #     os.makedirs(image_folder)

    # # Đếm số lượng ảnh hiện có trong thư mục
    # num_images = len([name for name in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, name))])

    # # Tạo tên tệp tin mới cho ảnh
    # image_filename = str(num_images + 1) + '.jpg'  # Định dạng tùy thuộc vào loại ảnh

    # # Lưu ảnh với tên mới vào thư mục
    # image_path = os.path.join(image_folder, image_filename)
    # with open(image_path, 'wb') as f:
    #     f.write(image_data)
    
    
    predicted_char, score = service.predict_image(image_data=image_data)

    return jsonify({'predicted_char': predicted_char, 'score': float(score)})

if __name__ == '__main__':
    app.run(debug=True)
