"""
bird classifier
"""
import os
import pdb
import time
import numpy as np
import constants
from image import Image
from model import Model
from labels import Labels


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow logging

DEFAULT_IMAGE_URLS = constants.IMAGE_URLS
DEFAULT_MODEL_URL = constants.MODEL_URL
DEFAULT_LABELS_URL = constants.LABELS_URL

class BirdClassifier:
    """ For classifying birds """
    def __init__(self, model, cleaned_labels):
        self.model = model
        self.cleaned_labels = cleaned_labels

    @staticmethod
    def fetch_or_load_model(self, model_url=DEFAULT_MODEL_URL):
        """ Load model if not already loaded in init """
        if self.model:
            return self.model
        else:
            bird_model = Model(model_url)
            return bird_model.load()

    @staticmethod
    def fetch_or_load_clean_labels(self, labels_url=DEFAULT_LABELS_URL):
        """ Load and cleanup labels if not already loaded and cleaned in init """
        if self.labels:
            return self.labels
            
        bird_labels = Labels(labels_url).load_and_cleanup()
        return bird_labels

    @classmethod
    def order_birds_by_result_score(cls, model_raw_output, bird_labels):
        """ Ordering results from model """
        for index, value in np.ndenumerate(model_raw_output):
            bird_index = index[1]
            bird_labels[bird_index]['score'] = value

        return sorted(bird_labels.items(), key=lambda x: x[1]['score'])

    @classmethod
    def get_top_n_result(cls, top_index, birds_names_with_results_ordered):
        """ Get top n results from ordered results """
        bird_name = birds_names_with_results_ordered[top_index*(-1)][1]['name']
        bird_score = birds_names_with_results_ordered[top_index*(-1)][1]['score']
        return bird_name, bird_score

    @classmethod
    def load_and_prep_image(cls, url):
        """ Fetch image and resize for model """
        image = Image(url)
        image_array = image.load()
        pre_processed_image = image.pre_process(image_array)
        return image.generate_image_tensor(pre_processed_image)

    @staticmethod
    def get_results_for_image(self, url):
        """ Get all results for a given image file """
        bird_model = self.fetch_or_load_model()
        bird_labels = self.fetch_or_load_clean_labels()
        image_tensor = self.load_and_prep_image(url)
        model_raw_output = bird_model.predict(image_tensor)
        birds_names_with_results_ordered = \
            self.order_birds_by_result_score(model_raw_output, bird_labels)
        return birds_names_with_results_ordered

    @staticmethod
    def get_results_for_images(self, urls):
        result_list = []
        for image_url in enumerate(urls):
            birds_names_with_results_ordered = self.get_results_for_image(image_url)
            result_list.append(birds_names_with_results_ordered)
        return result_list

    def print_results(self, index, birds_names_with_results_ordered):
        """ Print results to kubernetes log """
        print('Run: %s' % int(1 + index))
        bird_name, bird_score = self.get_top_n_result(1, birds_names_with_results_ordered)
        print('Top match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(2, birds_names_with_results_ordered)
        print('Second match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(3, birds_names_with_results_ordered)
        print('Third match: "%s" with score: %s' % (bird_name, bird_score))
        print('\n')


    def main(self):
        """ Main function """
        for index, image_url in enumerate(DEFAULT_IMAGE_URLS):
            birds_names_with_results_ordered = self.get_results_for_image(image_url)
            self.print_results(index, birds_names_with_results_ordered)

if __name__ == "__main__":
    start_time = time.time()
    classifier = BirdClassifier()
    classifier.main()
    print('Time spent: %s' % (time.time() - start_time))
