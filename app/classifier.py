"""
bird classifier
"""
import os
import time
import numpy as np
import constants
from image_processor import ImageProcessor
from model import Model
from labels import Labels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow logging

DEFAULT_IMAGE_URLS = constants.IMAGE_URLS
DEFAULT_MODEL_URL = constants.MODEL_URL
DEFAULT_LABELS_URL = constants.LABELS_URL


class BirdClassifier:
    """ For classifying birds """

    def __init__(self, model=None, cleaned_labels=None):
        self.model = model
        self.cleaned_labels = cleaned_labels

    def fetch_or_load_model(self, model_url=DEFAULT_MODEL_URL):
        """ Load model if not already loaded in init """
        if self.model:
            return self.model
        else:
            bird_model = Model(model_url)
            bird_model.load()
            bird_model.warmup()
            return bird_model

    def fetch_or_load_clean_labels(self, labels_url=DEFAULT_LABELS_URL):
        """ Load and cleanup labels if not already loaded and cleaned in init """
        if self.cleaned_labels:
            return self.cleaned_labels

        bird_labels = Labels(labels_url).load_and_cleanup()
        return bird_labels

    @classmethod
    def order_birds_by_result_score(cls, model_raw_output, bird_labels):
        """ Ordering results from model """
        for index, value in np.ndenumerate(model_raw_output):
            bird_index = index[1]
            bird_labels[bird_index]['score'] = value

        return sorted(bird_labels.items(), key=lambda x: x[1]['score'], reverse=True)

    @classmethod
    def get_top_n_result(cls, top_index, birds_names_with_results_ordered):
        """ Get top n results from ordered results """
        bird_name = birds_names_with_results_ordered[top_index][1]['name']
        bird_score = birds_names_with_results_ordered[top_index][1]['score']
        return bird_name, bird_score

    @classmethod
    def get_top_n_results(cls, ordered_results, n, minimum_score):
        top_n_results = ordered_results[:n]
        filtered_results = [result[1] for result in top_n_results if result[1]['score'] > minimum_score]
        return filtered_results

    def get_results_for_image(self, image_url):
        """ Get all results for a given image file """
        bird_model = self.fetch_or_load_model()
        bird_labels = self.fetch_or_load_clean_labels()
        image_tensor = ImageProcessor(image_url).load_and_prep_image()
        model_raw_output = bird_model.predict(image_tensor)
        birds_names_with_results_ordered = \
            self.order_birds_by_result_score(model_raw_output, bird_labels)
        return birds_names_with_results_ordered

    def get_results_for_bird_input(self, bird_input):
        result_list = {}
        for image_url in bird_input.image_urls:
            birds_names_with_results_ordered = \
                self.get_results_for_image(image_url)
            result_list[image_url.__str__()] = \
                self.get_top_n_results(
                    birds_names_with_results_ordered, bird_input.number_of_predictions, bird_input.minimum_score
                )
        return result_list

    def print_results(self, index, birds_names_with_results_ordered):
        """ Print results to kubernetes log """
        print('Run: %s' % int(1 + index))
        bird_name, bird_score = self.get_top_n_result(0, birds_names_with_results_ordered)
        print('Top match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(1, birds_names_with_results_ordered)
        print('Second match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(2, birds_names_with_results_ordered)
        print('Third match: "%s" with score: %s' % (bird_name, bird_score))
        print('\n')

    def main(self):
        """ Main function """
        for index, image_url in enumerate(DEFAULT_IMAGE_URLS):
            birds_names_with_results_ordered = self.get_results_for_image(image_url)
            self.print_results(index, birds_names_with_results_ordered)


if __name__ == "__main__":
    start_time = time.time()
    # Load the model
    model = Model(constants.MODEL_URL)
    model.load()

    # Warmup for faster prediction
    model.warmup()

    # Load and clean labels
    labels = Labels(constants.LABELS_URL)
    cleaned_labels = labels.load_and_cleanup()

    classifier = BirdClassifier(model, cleaned_labels)
    classifier.main()
    print('Time spent: %s' % (time.time() - start_time))
