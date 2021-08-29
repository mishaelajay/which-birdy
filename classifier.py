"""
bird classifier
"""
import os
import urllib.request
import time
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import constants


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow logging

IMAGE_URLS = constants.IMAGE_URLS
MODEL_URL = constants.MODEL_URL
LABELS_URL = constants.LABELS_URL

"""
bird classifier
"""
class BirdClassifier:
    @staticmethod
    def load_model():
        return hub.KerasLayer(MODEL_URL)

    def load_and_cleanup_labels(self):
        # bird_labels_raw = urllib.request.urlopen(LABELS_URL)
        with urllib.request.urlopen(LABELS_URL) as bird_labels_raw:
            bird_labels_lines = self.fetch_lines_from_label(bird_labels_raw)
            bird_labels_lines.pop(0)
            birds = {}
            for bird_line in bird_labels_lines:
                bird_id = int(bird_line.split(',')[0])
                bird_name = bird_line.split(',')[1]
                birds[bird_id] = {'name': bird_name}
        return birds

    @classmethod
    def fetch_lines_from_label(cls, bird_labels_raw):
        return [line.decode('utf-8').replace('\n', '') for line in bird_labels_raw.readlines()]

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

    def get_results_for_file(self, url):
        bird_model = self.load_model()
        bird_labels = self.load_and_cleanup_labels()
        with urllib.request.urlopen(url) as image_get_response:
            image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            image_tensor = tf.expand_dims(image_tensor, 0)
            model_raw_output = bird_model.call(image_tensor).numpy()
            birds_names_with_results_ordered = \
                self.order_birds_by_result_score(model_raw_output, bird_labels)
        return birds_names_with_results_ordered

    def main(self):
        for index, image_url in enumerate(IMAGE_URLS):
            bird_model = self.load_model()
            bird_labels = self.load_and_cleanup_labels()
            # Loading images
            with urllib.request.urlopen(image_url) as image_get_response:
                image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
                # Changing images
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255
                # Generate tensor
                image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
                image_tensor = tf.expand_dims(image_tensor, 0)
                model_raw_output = bird_model.call(image_tensor).numpy()
            birds_names_with_results_ordered = \
                self.order_birds_by_result_score(model_raw_output, bird_labels)
            # Print results to kubernetes log
            print('Run: %s' % int(index + 1))
            bird_name, bird_score = self.get_top_n_result(1, birds_names_with_results_ordered)
            print('Top match: "%s" with score: %s' % (bird_name, bird_score))
            bird_name, bird_score = self.get_top_n_result(2, birds_names_with_results_ordered)
            print('Second match: "%s" with score: %s' % (bird_name, bird_score))
            bird_name, bird_score = self.get_top_n_result(3, birds_names_with_results_ordered)
            print('Third match: "%s" with score: %s' % (bird_name, bird_score))
            print('\n')


if __name__ == "__main__":
    start_time = time.time()
    classifier = BirdClassifier()
    classifier.main()
    print('Time spent: %s' % (time.time() - start_time))
