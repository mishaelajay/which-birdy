"""
A class to perform Image related operations before prediction by model
"""
import urllib.request
import cv2
import numpy as np
import tensorflow.compat.v2 as tf


class ImageProcessor:
    """ high level support for performing image related operations """

    def __init__(self, source):
        self.source = source

    def load(self):
        """ Read image from url """
        with urllib.request.urlopen(self.source) as image_get_response:
            return np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)

    @classmethod
    def pre_process(cls, image_array):
        """ Resize and modify image for model to process """
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image / 255

    @classmethod
    def generate_image_tensor(cls, pre_processed_image):
        """ generate image tensor for the model to process """
        image_tensor = tf.convert_to_tensor(pre_processed_image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        return image_tensor

    def load_and_prep_image(self):
        """ Fetch image and resize for model """
        image_array = self.load()
        pre_processed_image = self.pre_process(image_array)
        return self.generate_image_tensor(pre_processed_image)
