from flask import Flask, Response
from predictor import Predictor

app = Flask(__name__)

def predict():
    return Predictor().predict()
@app.route('/')
def home():
    pred = predict()
    return pred
 
if __name__ == '__main__': 
    app.run()
