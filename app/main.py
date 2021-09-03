from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from classifier import BirdClassifier
import uvicorn
import pdb
from model import Model
from labels import Labels
import constants

# Create the app object
app = FastAPI()

# Load the model
model = Model(constants.MODEL_URL)
model.load()

# Load and clean labels
labels = Labels(constants.LABELS_URL)
cleaned_labels = labels.load_and_cleanup()

# Load the main classifier withmodel and labels
bird_classifier = BirdClassifier(model, cleaned_labels)

# input class for image urlss
class BirdInput(BaseModel):
    image_urls: Optional[List[str]] = None
    number_of_predictions: Optional[int] = 3
    precision: Optional[float] = 0.0

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Bird enthusiast. Checkout our docs page by appending /docs to the current url'}

# Takes image url as input and returns top 3 results as output
@app.post('/which-bird')
async def classify_bird(images: List[BirdInput]):
    ordered_results = bird_classifier.get_results_for_images(images.image_urls)
    return {}

def form_result_dict(ordered_results):
    return {}

if __name__ == "__main__":
    uvicorn.run(app, debug=True)