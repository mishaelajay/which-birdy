""" Model class to handle model related operations """
import tensorflow_hub as hub

class Model:
    """ Model class to handle model related operations """
    def __init__(self, model_url):
        self.model_url = model_url
        self.loaded_model = None

    def load(self):
        """ Load model using model url """
        self.loaded_model = hub.KerasLayer(self.model_url)
        return self.loaded_model

    def predict(self, image_tensor):
        """ Predict image tensor """
        return self.loaded_model.call(image_tensor).numpy()
