from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
logistic_model = pickle.load(open("car_price.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = logistic_model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Your are Covid(positive=1/negetive=0): {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)