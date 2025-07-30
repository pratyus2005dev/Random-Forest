# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    level = np.array([[data['level']]])
    prediction = model.predict(level)[0]
    return jsonify({'predicted_salary': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
