from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from classifier import BirdClassifier
import time
from model import Model
from labels import Labels
from image_processor import ImageProcessor
import constants
import uvicorn

# Create the app object
app = FastAPI()


class BirdInput(BaseModel):
    """" Input class for image urls"""
    image_urls: List[HttpUrl] = None
    number_of_predictions: Optional[int] = 3
    minimum_score: Optional[float] = 0.0


@app.on_event("startup")
async def startup_event():
    """ Preload the classifier with model and labels """
    model = preload_model()
    cleaned_labels = preload_labels()
    app.bird_classifier = BirdClassifier(model, cleaned_labels)


# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Bird enthusiast. Checkout our docs page by appending /docs to the current url'}


# Takes image url as input and returns top 3 results as output
@app.post('/which-bird')
async def classify_bird(bird_input: BirdInput):
    start_time = time.time()
    validate_urls(bird_input.image_urls)
    ordered_results = app.bird_classifier.get_results_for_bird_input(bird_input)
    ordered_results['time_taken'] = time.time() - start_time
    return str(ordered_results)


def preload_model():
    """ Load the model """
    model = Model(constants.MODEL_URL)
    model.load_and_warmup()
    return model


def preload_labels():
    """ Load and clean labels """
    labels = Labels(constants.LABELS_URL)
    return labels.load_and_cleanup()


def validate_urls(image_urls):
    """ Raise HTTPException when a url is invalid """
    for url in image_urls:
        if not (ImageProcessor.is_url(url)):
            raise HTTPException(status_code=422, detail='One of the urls you entered is invalid: %s' % url)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
