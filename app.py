from flask import Flask, Response
from predictor import Predictor

app = Flask(__name__)

def predict():
    return Predictor().predict()
@app.route('/')
def home():
    def generate():
      for i in range(100):
        yield "<br/>"  
        yield str(predict())
    return Response(generate(), mimetype='text/html')
 
if __name__ == '__main__': 
    app.run()
