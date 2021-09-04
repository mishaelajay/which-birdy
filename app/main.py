from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from classifier import BirdClassifier
import uvicorn
import time
from model import Model
from labels import Labels
import constants

# TODO: if __name__ == "__main__"

# Create the app object
app = FastAPI()

# TODO: put in function
# Load the model
model = Model(constants.MODEL_URL)
model.load()
model.warmup()


# TODO: in a function
# Load and clean labels
labels = Labels(constants.LABELS_URL)
cleaned_labels = labels.load_and_cleanup()

# Load the main classifier with model and labels
bird_classifier = BirdClassifier(model, cleaned_labels)


# input class for image urls
class BirdInput(BaseModel):
    image_urls: Optional[List[str]] = None
    number_of_predictions: Optional[int] = 3
    minimum_score: Optional[float] = 0.0


# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Bird enthusiast. Checkout our docs page by appending /docs to the current url'}


# Takes image url as input and returns top 3 results as output
@app.post('/which-bird')
async def classify_bird(bird_input: BirdInput):
    start_time = time.time()
    ordered_results = bird_classifier.get_results_for_bird_input(bird_input)
    ordered_results['time_taken'] = time.time() - start_time
    return str(ordered_results)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
