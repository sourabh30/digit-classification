import joblib
from flask import Flask, request, jsonify
# from utilities.utilities import preprocess, get_ml_model
from sklearn.preprocessing import StandardScaler
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

@app.route('/predict/<model_type>', methods=['POST'])
def load_model(model_type):

    supported_model_types = ['svm', 'tree', 'lr']
    
    if model_type not in supported_model_types:
        return { "model_type" : f"{model_type} model not supported. Supported models {supported_model_types}"}

    # Get the image array from the input
    image_array = get_image_array()

    # Comvert string image array to np array
    image_array = get_np_image_array(image_array)

    # Pre processing the image
    preprocessed_image = preprocess(image_array)

    # Get the model to predict
    best_model = get_model_by_type(model_type)

    # Use the loaded model for prediction
    predicted_digit = best_model.predict(preprocessed_image.reshape(1, -1))[0]

    # Generate and send the predicated data
    return generate_response(predicted_digit)


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

def get_model_by_type(model_type):
    if model_type == "svm":
        return joblib.load(r"./models/M22AIE249_best_svm_model_svm_C_5_gamma_0.01.pkl")
    elif model_type == "tree":
        return joblib.load(r"./models/M22AIE249_best_decision_tree_model_decision_tree_max_depth_20_min_samples_split_2.pkl")
    elif model_type == "lr":
        return joblib.load(r"./models/M22AIE249_best_logistic_regression_model_logistic_regression_solver_newton-cg.pkl")


def generate_response(predicted_digit):
    response = {
        "predicted_digit": int(predicted_digit)
    }
    return jsonify(response)

def preprocess(x):
    num_samples = len(x)
    x = x.reshape((num_samples, -1))

    # Applying unit normalization using StandardScaler
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(x)

    return data_normalized

def get_ml_model(dir):
    # Dynamically load the first model in the 'models/' folder
    model_files = os.listdir(f'{dir}/')
    # model_files = os.listdir('models/')
    model_files = [file for file in model_files if file.endswith('.pkl')]

    if not model_files:
        raise FileNotFoundError("No model files found in the 'models/' folder")

    # try:
    #     first_model_file = model_files[3]
    # except:
    #     first_model_file = model_files[0]

    first_model_file = model_files[0]
    first_model_path = f"models/{first_model_file}"
    return first_model_path


if __name__ == "__main__":
    app.run(debug=True)