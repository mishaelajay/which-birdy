from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from classifier import BirdClassifier
import uvicorn
import time
from model import Model
from labels import Labels
import constants

# Create the app object
app = FastAPI()


# input class for image urls
class BirdInput(BaseModel):
    image_urls: Optional[List[str]] = None
    number_of_predictions: Optional[int] = 3
    minimum_score: Optional[float] = 0.0


@app.on_event("startup")
async def startup_event():
    # Preload the classifier with model and labels
    model = preload_model()
    cleaned_labels = preload_labels()
    app.bird_classifier = BirdClassifier(model, cleaned_labels)


def preload_model():
    # Load the model
    model = Model(constants.MODEL_URL)
    model.load_and_warmup()
    return model


def preload_labels():
    # Load and clean labels
    labels = Labels(constants.LABELS_URL)
    return labels.load_and_cleanup()


# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Bird enthusiast. Checkout our docs page by appending /docs to the current url'}


# Takes image url as input and returns top 3 results as output
@app.post('/which-bird')
async def classify_bird(bird_input: BirdInput):
    start_time = time.time()
    ordered_results = app.bird_classifier.get_results_for_bird_input(bird_input)
    ordered_results['time_taken'] = time.time() - start_time
    return str(ordered_results)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
