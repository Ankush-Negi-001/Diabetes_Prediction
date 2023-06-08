from flask import Flask, request, url_for, jsonify, redirect, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

dataset = pd.read_csv('Diabetes_dataset.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods= ['POST', 'GET'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."

    return render_template('index.html', prediction_text='{}'.format(pred))

if __name__ == '__main__':
    app.run(debug=True) 