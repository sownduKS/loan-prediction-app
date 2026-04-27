from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app) # This allows your website to talk to this script

# Load the brain we just trained
model = joblib.load('loan_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the website
    data = request.json
    
    # Organize data for the model
    features = np.array([[data['income'], data['loan'], data['credit']]])
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Send result back
    result = "Approved" if prediction[0] == 1 else "Rejected"
    return jsonify({'status': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)