from flask import Flask
from predictor import Predictor

app = Flask(__name__)

def predict():
    return Predictor().predict()
@app.route('/')
def home():
    print('Predicting')
    return '<h1> Corona Predictor </h1>', predict()
 
if __name__ == '__main__': 
    app.run()
