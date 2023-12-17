from flask import Flask
from flask import request
from flask import jsonify
import sklearn
import pickle
import pandas as pd

app = Flask("cardio_concern")
dv, modellr = pickle.load(open("modellr.pkl", "rb"))


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = modellr.predict_proba(X)[0, 1]
    cardio_concern = y_pred >= 0.5

    result = {
        'cardio_status': float(y_pred),
        'cardio_concern': int(cardio_concern)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
