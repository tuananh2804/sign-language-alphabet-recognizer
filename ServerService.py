import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
import base64

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

        predictions = self.sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})
        
        nparr = np.frombuffer(image_data, np.uint8)
        img_fliped = cv2.flip(nparr, 1)
        img = cv2.imdecode(img_fliped, cv2.IMREAD_COLOR)

        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]

        image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()

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