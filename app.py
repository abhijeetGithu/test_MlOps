from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
# pylint: disable=no-member
target_names = iris.target_names

# Load the trained model
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for rendering the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the request JSON
    request_data = request.get_json()
    if 'features' not in request_data:
        return jsonify({'error': 'Missing features in request'}), 400
    
    features = request_data['features']
    
    # Convert features to numpy array
    try:
        features = np.array(features).reshape(1, -1)
    except ValueError:
        return jsonify({'error': 'Invalid features provided'}), 400
    
    # Make prediction
    prediction = model.predict(features)
    
    # Map prediction index to class label
    class_label = target_names[prediction[0]]
    
    # Return the prediction
    return jsonify({'prediction': class_label})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
