
import tensorflow_hub as hub

class Model:
    """Model class to handle model related operations"""
    def __init__(self, model_url):
       self.loaded_model = hub.KerasLayer(model_url)