import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
import base64
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf

class Service:
    def __init__(self):
        self.label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]
        self.sess = tf.Session()
        with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def predict_image(self, image_data):

        softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, -1)
        # img = cv2.flip(img, 1)
        
        # # Lưu ảnh đã cắt vào thư mục "image_cropped"
        # cropped_folder = 'image_cropped'
        # if not os.path.exists(cropped_folder):
        #     os.makedirs(cropped_folder)

        # # Đếm số lượng ảnh đã cắt trong thư mục
        # num_cropped_images = len([name for name in os.listdir(cropped_folder) if os.path.isfile(os.path.join(cropped_folder, name))])

        # # Tạo tên tệp tin mới cho ảnh đã cắt
        # cropped_image_filename = str(num_cropped_images + 1) + '.jpg'

        # # Lưu ảnh đã cắt với tên mới vào thư mục
        # cropped_image_path = os.path.join(cropped_folder, cropped_image_filename)
        # cv2.imwrite(cropped_image_path, img)

        image_data = cv2.imencode('.jpg', img)[1].tostring()
        
        predictions = self.sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        max_score = 0.0
        res = ''
        for node_id in top_k:
            human_string = self.label_lines[node_id]
            score = predictions[0][node_id]
            if score > max_score:
                max_score = score
                res = human_string
        return res, max_score

    def close_session(self):
        self.sess.close()