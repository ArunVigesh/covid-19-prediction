from flask import Flask, Response
from predictor import Predictor

app = Flask(__name__)

def predict():
    return Predictor().predict()
@app.route('/')
def home():
    pred = predict()
    i = 10000000000000000000000
    while i>=0:
        i = i-1
    return pred
 
if __name__ == '__main__': 
    app.run()
