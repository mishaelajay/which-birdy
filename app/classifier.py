"""
bird classifier
"""
import os
from re import L
import urllib.request
import time
import numpy as np
import constants
from image import Image
from model import Model
from labels import Labels


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow logging

IMAGE_URLS = constants.IMAGE_URLS
MODEL_URL = constants.MODEL_URL
LABELS_URL = constants.LABELS_URL

"""
bird classifier
"""
class BirdClassifier:
    @classmethod
    def order_birds_by_result_score(cls, model_raw_output, bird_labels):
        for index, value in np.ndenumerate(model_raw_output):
            bird_index = index[1]
            bird_labels[bird_index]['score'] = value

        return sorted(bird_labels.items(), key=lambda x: x[1]['score'])

    @classmethod
    def get_top_n_result(cls, top_index, birds_names_with_results_ordered):
        bird_name = birds_names_with_results_ordered[top_index*(-1)][1]['name']
        bird_score = birds_names_with_results_ordered[top_index*(-1)][1]['score']
        return bird_name, bird_score

    @classmethod
    def load_model_and_clean_labels(self, model_url, labels_url):
        bird_model = Model(MODEL_URL).loaded_model
        bird_labels = Labels(LABELS_URL).load_and_cleanup()
        return bird_model, bird_labels

    @classmethod
    def load_and_prep_image(self, url):
        image = Image(url)
        image_array = image.load()
        pre_processed_image = image.pre_process(image_array)
        return image.generate_image_tensor(pre_processed_image)

    def get_results_for_file(self, url):
        bird_model, bird_labels = self.load_model_and_clean_labels(MODEL_URL, LABELS_URL)
        image_tensor = self.load_and_prep_image(url)
        model_raw_output = bird_model.call(image_tensor).numpy()
        birds_names_with_results_ordered = \
            self.order_birds_by_result_score(model_raw_output, bird_labels)
        return birds_names_with_results_ordered

    def print_results(self, index, birds_names_with_results_ordered):
        # Print results to kubernetes log
        print('Run: %s' % int(1 + index))
        bird_name, bird_score = self.get_top_n_result(1, birds_names_with_results_ordered)
        print('Top match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(2, birds_names_with_results_ordered)
        print('Second match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(3, birds_names_with_results_ordered)
        print('Third match: "%s" with score: %s' % (bird_name, bird_score))
        print('\n')


    def main(self):
        for index, image_url in enumerate(IMAGE_URLS):
            birds_names_with_results_ordered = self.get_results_for_file(image_url)
            self.print_results(index, birds_names_with_results_ordered)

if __name__ == "__main__":
    start_time = time.time()
    classifier = BirdClassifier()
    classifier.main()
    print('Time spent: %s' % (time.time() - start_time))
