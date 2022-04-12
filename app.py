from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('./models/model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['cylinders']
    val2 = request.form['displacement']
    val3 = request.form['horsepower']
    val4 = request.form['weight']
    val5 = request.form['acceleration']
    val6 = request.form['year']
    arr = np.array([val1, val2, val3, val4, val5, val6])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=round(float(pred),2))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
