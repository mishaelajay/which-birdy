# which-birdy

Bird classifier 

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

Please refer to ```deployment.yaml``` and ```service.yaml``` in kubernetes folder for kubectl deployment details. I can show you my gcp dashboard as well. 


