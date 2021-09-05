""" Model class to handle model related operations """
import tensorflow_hub as hub
from image_processor import ImageProcessor

WARMUP_IMAGE_PATH = 'app/tests/images/Eumomota_superciliosa.jpeg'


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
        """ Warming up model for faster results """
        warmup_image_tensor = self.warmup_image_tensor()
        self.predict(warmup_image_tensor)

    @staticmethod
    def warmup_image_tensor():
        """ Method to fetch and generate warmup image tensor """
        image_processor = ImageProcessor(WARMUP_IMAGE_PATH)
        return image_processor.load_and_prep_image()

    def predict(self, image_tensor):
        """ Predict image tensor """
        return self.loaded_model.call(image_tensor).numpy()

    def load_and_warmup(self):
        """ Return model after loading and warmup """
        self.load()
        self.warmup()
        return self.loaded_model
