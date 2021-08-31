from fastapi import FastAPI, File, UploadFile
from typing import List
from classifier import BirdClassifier

# Create the app object
app = FastAPI()

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Bird enthusiast'}

# Takes image url as input and returns top 3 results as output
@app.get('/which-bird/image')
async def classify_bird(image_urls: List[str]):
    BC = BirdClassifier()
    ordered_results = BC.get_results_for_file(image_urls[0])
    return {}
