import urllib.request

class Labels:
    """Labels class to load and read labels"""

    def __init__(self, labels_url):
        self.labels_url = labels_url

    def load_and_cleanup(self):
        with urllib.request.urlopen(self.labels_url) as bird_labels_raw:
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
