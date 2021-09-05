# which-birdy

Bird classifier 

Hi all, 

I had very limited time due to existing work commitments. But i did what i could. Not exactly my complete vision of nice code.
I have listed the possible improvements i could make given more time.

Uses fast api.

Please use [my docker image](https://hub.docker.com/repository/docker/mishaelajay/which-birdy) for easy deployment on your local.

You can also alternatively run 

```
python app/main.py
```

To run the classifier directly 

```
python app/classifier.py
```

I have also deployed it using Google Cloud Platform and Kubernetes on 34.136.130.19.

A sample curl request using the deployed api:

```
curl -X 'POST' \
  'http://34.136.130.19/which-bird' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_urls": [
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg",
    "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg",
    "https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg",
    "https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg"
  ],
  "number_of_predictions": 3,
  "minimum_score": 0
}'
```

Sample response:
```
{
  'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg':
    [{'name': 'Phalacrocorax varius varius', 'score': 0.000158282}, {'name': 'Phalacrocorax varius', 'score': 0.00033242477}, {'name': 'Microcarbo    
    melanoleucos','score': 9.651384e-05}],
  'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg': 
    [{'name': 'Galerida cristata', 'score': 5.419277e-05}, {'name': 'Alauda arvensis', 'score': 4.9903865e-05}, {'name': 'Eremophila alpestris', 'score':                6.390799e05}], 
  'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg': 
    [{'name': 'Eumomota superciliosa', 'score': 9.651384e-05}, {'name': 'Momotus coeruliceps', 'score': 0.000158282}, {'name': 'Momotus lessonii', 'score':             0.000158282}],
  'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg': 
    [{'name': 'Aulacorhynchus prasinus', 'score': 3.8968494e-05}, {'name': 'Cyanocorax yncas', 'score': 0.000104808554}, {'name': 'Chlorophanes spiza', 'score':         0.00018665729}], 
  'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg': 
    [{'name': 'Erithacus rubecula', 'score': 0.8382054}, {'name': 'Ixoreus naevius', 'score': 0.0030795017}, {'name': 'Setophaga tigrina', 'score': 0.002611359}],
  'time_taken': 2.556856393814087
 }
```

Improvments done:
- Improved time taken by ```classifier.py``` by warming up model
- Added necessary abstraction by separating code into different classes.
- Served the model using fastapi. Added validations and loaded model, cleaned labels only once at startup.
- Improved time taken by the endpoint to send back results.
- Added input fields to pass in number of predictions and minimum score.


Further improvements (Can be done with more time):
- Test cases for each of the classes and the fast api endpoint.
- A separate microservice for the model and the uvicorn server. So that we can make maximum use of the Kubernetes LoadBalancer to switch between clusters based on need.
- A better customised docker image instead of using https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker.
- Further validation of input and output.
- Better error handling and logging.

Please refer to ```deployment.yaml``` and ```service.yaml``` in kubernetes folder for kubectl deployment details. I can show you my gcp dashboard as well. 

Looking forward to your feedback.

TIA
