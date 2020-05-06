from flask import Flask
from predictor import Predictor

app = Flask(__name__)

@app.route('/')
def home():
    return Predictor().predict()
 
if __name__ == '__main__': 
    app.run()
