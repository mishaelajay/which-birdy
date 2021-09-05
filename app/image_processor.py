"""
A class to perform Image related operations before prediction by model
"""
import urllib.request
import cv2
import numpy as np
import tensorflow.compat.v2 as tf
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image


class ImageProcessor:
    """ high level support for performing image related operations """

    def __init__(self, source):
        self.source = source
        self.loaded_image = None

    def load(self):
        """ Load based on type of source entered """
        if self.loaded_image:
            return self.loaded_image
        else:
            if self.is_url(self.source):
                return self.load_from_url()
            elif self.is_file_path():
                return self.load_from_path()
            else:
                raise ValueError('Source is neither valid url nor path')

    def load_from_url(self):
        """ Read image from url """
        try:
            with urllib.request.urlopen(self.source) as image_get_response:
                image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
                # return image after colour correction
                return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except urllib.error.HTTPError as e:
            print('Could not open Image')
            print('Error details %s' % str(e))

    def load_from_path(self):
        """ Read image from path """
        image = Image.open(self.source)
        return np.asarray(image)

    def is_file_path(self):
        """ Check if source is a valid file path"""
        return Path(self.source).is_file()

    @classmethod
    def is_url(cls, url):
        """ Check if url is valid before loading """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @classmethod
    def pre_process(cls, image_array):
        """ Resize and modify image for model to process """
        image = cv2.resize(image_array, (224, 224))
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
