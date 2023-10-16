import pickle
import requests ## to use the POST method we use a library named requests
from flask import FLASK
from flask import request
from flask import jsonify

with open('model1.bin', 'rb') as f_in:
     model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

case = {"job": "retired", "duration": 445, "poutcome": "success"}
def predict(case):
    X = dv.transform([case])
    y_p = model.predict_proba(X)
    y_p = y_p[:, 1]
    return y_p[0]

print(predict(case))



#----------------------------------------------------------------------------------------#

app = FLASK('predict')

@app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
def predict():
    customer = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.

    prediction = predict(customer)
    positive = prediction >= 0.5

    result = {
        'positive_probability': float(prediction), ## we need to cast numpy float type to python native float type
        'positive': bool(positive),  ## same as the line above, casting the value using bool method
    }

    return jsonify(result)

customer = {"job": "unknown", "duration": 270, "poutcome": "failure"}
url = 'http://localhost:9696/predict' ## this is the route we made for prediction
response = requests.post(url, json=customer) ## post the customer information in json format
result = response.json() ## get the server response
print(result)
