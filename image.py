import cv2
import numpy as np
import urllib.request
from numpy.lib.type_check import imag
import tensorflow.compat.v2 as tf

class Image:
    def __init__(self, url):
        self.image_url = url
        self.predictable_image_tensor = None

    def load(self):
        image_get_response = urllib.request.urlopen(self.image_url)
        return np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)

    def pre_process(self, image_array):
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image / 255
    
    def generate_image_tensor(self, pre_processed_image):
        image_tensor = tf.convert_to_tensor(pre_processed_image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        self.predictable_image_tensor = image_tensor
        return image_tensor