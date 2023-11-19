import pytest
from utilities import create_hparam_combo
from sklearn import datasets
import numpy as np
from api.app import app

# def test_create_hparam_combo():
#     gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
#     C_range = [0.1, 1.0, 2, 5, 10]
#     param_combinations = create_hparam_combo(gamma_range, C_range)
#     assert len(param_combinations) == len(gamma_range) * len(C_range)


def test_post_predict():
    PREDICTED_DIGIT = "predicted_digit"

    # Function to call the predict api
    response = predict_digit(0)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 0

    # Function to call the predict api
    response = predict_digit(1)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 1

    # Function to call the predict api
    response = predict_digit(2)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 2

    # Function to call the predict api
    response = predict_digit(3)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 3

    # Function to call the predict api
    response = predict_digit(4)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 4

    # Function to call the predict api
    response = predict_digit(5)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 5

    # Function to call the predict api
    response = predict_digit(6)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 6

    # Function to call the predict api
    response = predict_digit(7)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 7

    # Function to call the predict api
    response = predict_digit(8)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 8

    # Function to call the predict api
    response = predict_digit(9)
    assert response.get_json()[f'{PREDICTED_DIGIT}'] == 9

    # Assert the status_code
    assert response.status_code == 200


def predict_digit(digit):
    ENDPOINT = "predict"

    # Generate the np input array for digits
    json_image_array = generate_np_array_for_digit(digit)

    # generate the input req for predict api
    input_request = generate_input_request(json_image_array)

    # make the post call to predict api
    response = make_post_call(
        endpoint=ENDPOINT,
        request=input_request
    )

    return response


def get_predicted_data(response):
    return response.get_json()['predicted_digit']


def make_post_call(endpoint, request):
    return app.test_client().post(
        f"/{endpoint}",
        json=request
    )


def generate_input_request(json_image_array):
    return {
        "image": f"{json_image_array}"
    }


def generate_np_array_for_digit(num):
    generate_arry_for_digit = num

    # Load the digits dataset
    digits = datasets.load_digits()

    # Select samples for digit
    digit_samples = digits.data[digits.target == generate_arry_for_digit]

    # Take a random sample from the selected digit samples
    random_index = np.random.randint(0, digit_samples.shape[0])
    random_digit = digit_samples[random_index]

    # Convert the NumPy array to a JSON-formatted string
    json_image_array = random_digit.tolist()
    return json_image_array


# test_post_predict()

# from utilities import get_hyperparameter_combinations, train_test_dev_split,read_digits, tune_hparams, preprocess_data
# import os

# def test_for_hparam_cominations_count():
#     # a test case to check that all possible combinations of paramers are indeed generated
#     gamma_list = [0.001, 0.01, 0.1, 1]
#     C_list = [1, 10, 100, 1000]
#     h_params={}
#     h_params['gamma'] = gamma_list
#     h_params['C'] = C_list
#     h_params_combinations = get_hyperparameter_combinations(h_params)

#     assert len(h_params_combinations) == len(gamma_list) * len(C_list)

# def create_dummy_hyperparameter():
#     gamma_list = [0.001, 0.01]
#     C_list = [1]
#     h_params={}
#     h_params['gamma'] = gamma_list
#     h_params['C'] = C_list
#     h_params_combinations = get_hyperparameter_combinations(h_params)
#     return h_params_combinations
# def create_dummy_data():
#     X, y = read_digits()

#     X_train = X[:100,:,:]
#     y_train = y[:100]
#     X_dev = X[:50,:,:]
#     y_dev = y[:50]

#     X_train = preprocess_data(X_train)
#     X_dev = preprocess_data(X_dev)

#     return X_train, y_train, X_dev, y_dev
# def test_for_hparam_cominations_values():
#     h_params_combinations = create_dummy_hyperparameter()

#     expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
#     expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

#     assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

# def test_model_saving():
#     X_train, y_train, X_dev, y_dev = create_dummy_data()
#     h_params_combinations = create_dummy_hyperparameter()

#     _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev,
#         y_dev, h_params_combinations)

#     assert os.path.exists(best_model_path)

# def test_data_splitting():
#     X, y = read_digits()

#     X = X[:100,:,:]
#     y = y[:100]

#     test_size = .1
#     dev_size = .6

#     X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

#     assert (len(X_train) == 30)
#     assert (len(X_test) == 10)
#     assert  ((len(X_dev) == 60))
