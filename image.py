"""
A class to perform Image related operations before prediction by model
"""
import urllib.request
import cv2
import numpy as np
import tensorflow.compat.v2 as tf

class Image:
    """ high level support for performing image related operations """
    def __init__(self, url):
        self.image_url = url
        self.predictable_image_tensor = None

    def load(self):
        """ Read image from url """
        with urllib.request.urlopen(self.image_url) as image_get_response:
            return np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)

    @classmethod
    def pre_process(cls, image_array):
        """ Resize and modify image for model to process """
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image / 255

    def generate_image_tensor(self, pre_processed_image):
        """ generate image tensor for the model to process """
        image_tensor = tf.convert_to_tensor(pre_processed_image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        self.predictable_image_tensor = image_tensor
        return image_tensor
