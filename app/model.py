""" Model class to handle model related operations """
import PIL
import cv2
import tensorflow_hub as hub
import numpy as np
from image_processor import ImageProcessor


class Model:
    """ Model class to handle model related operations """

    def __init__(self, model_url):
        self.model_url = model_url
        self.loaded_model = None

    def load(self):
        """ Load model using model url """
        self.loaded_model = hub.KerasLayer(self.model_url)
        return self.loaded_model

    def warmup(self):
        warmup_image = self.warmup_image()
        image_tensor = ImageProcessor.generate_image_tensor(warmup_image)
        self.predict(image_tensor)

    @staticmethod
    def warmup_image():
        image = PIL.Image.open('tests/images/Eumomota_superciliosa.jpeg')
        image_array = np.asarray(image)
        image = cv2.resize(image_array, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image / 255

    def predict(self, image_tensor):
        """ Predict image tensor """
        return self.loaded_model.call(image_tensor).numpy()
