import joblib
from flask import Flask, request, jsonify
from ..utilities import preprocess, get_ml_model
import os
import numpy as np
import json

from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return f'User {escape(username)}'


@app.route('/sum/<x>/<y>')
def sum_two_num(x,y):
    return f'Sum of {x} and {y} = {int(x) + int(y)}'

@app.route('/sum_int/<int:x>/<int:y>')
def sum_two_num_int(x,y):
    return f'Sum of two interger {x} and {y} = {x + y}'

@app.route('/sum_numbers', methods = ['POST'])
def sum_num():
    data = request.get_json()
    x = data['x']
    y = data['y']
    sum = int(x) + int(y)
    print(sum)
    return str(sum)


@app.route('/compare_images', methods=['POST'])
def compare_images():
    data = request.get_json()
    # Get the image files from the request
    image1_array = data['image1']
    image2_array = data['image2']

    # Compare the NumPy arrays
    if image1_array == image2_array:
        status = True
    else:
        status = False

    # status = True
    response = {
        "result": status 
    }

    return response


@app.route('/predict', methods=['POST'])
def predict_digit():
    
    # Get the image array from the input
    image_array = get_image_array()

    # Comvert string image array to np array
    image_array = get_np_image_array(image_array)

    # Pre processing the image
    preprocessed_image = preprocess(image_array)

    # Get the best model to predict
    best_model = get_best_model()

    # Use the loaded model for prediction
    predicted_digit = best_model.predict(preprocessed_image.reshape(1, -1))[0]

    # Generate and send the predicated data
    return generate_response(predicted_digit)


def get_image_array():
    data = request.get_json()
    return data['image']

def get_np_image_array(image_array):
    return np.array(json.loads(image_array))

def get_best_model():
    return joblib.load(get_ml_model('models'))

def generate_response(predicted_digit):
    response = {
        "predicted_digit": int(predicted_digit)
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)