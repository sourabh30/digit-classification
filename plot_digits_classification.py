# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

import itertools
from sklearn import datasets, metrics, svm
from utilities import train_model, predict_and_eval, split_train_dev_test, preprocess, tune_hyperparameters, create_hparam_combo

# 1. Get the dataset
digits = datasets.load_digits()
x = digits.images
y = digits.target

# Define the list of dev and test sizes
dev_sizes = [0.1, 0.2, 0.3]
test_sizes = [0.1, 0.2, 0.3]

# Create combinations using itertools.product
dev_test_combinations = [{'test_size': test, 'dev_size': dev} for test, dev in itertools.product(test_sizes, dev_sizes)]


for dev_test in dev_test_combinations:
    test_size = dev_test['test_size']
    dev_size = dev_test['dev_size']
    train_size = 1 - (dev_size+test_size)

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=0.2, dev_size=0.25);

    # 3. Data preprocessing 
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    X_dev = preprocess(X_dev)

    gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
    C_range = [0.1, 1.0, 2, 5, 10]

    # Generate a list of dictionaries representing all combinations
    # param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in itertools.product(gamma_range, C_range)]
    param_combinations = create_hparam_combo(gamma_range, C_range)

    # Hyperparameter tuning 
    train_acc, best_hparams, best_model, best_accuracy = tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations)

    # Train the data
    result = train_model(X_train, y_train, {'gamma': 0.001}, model_type='svm')

    # Accuracy Evaluation
    accuracy_test = predict_and_eval(result,X_test, y_test)

    # print("Accuracy on Test Set:", accuracy_test)
    # print("Classification Report on Test Set:\n", classification_rep_test)
    # print("Confusion Matrix on Test Set:\n", confusion_mat_test)

    # Print all combinations 
    print(f'test_size={test_size}, dev_size={dev_size}, train_size={train_size}, train_acc:{train_acc} dev_acc:{best_accuracy} test_acc: {accuracy_test}')
    print(f' Best params:{best_hparams}')


# Added as part of test 2.1
print("Length of data set:", len(x))
# Added as part of test 2.2
for image in x:
    height, width = image.shape[0], image.shape[1]
    print(f"Image Size - Width: {width}, Height: {height}")