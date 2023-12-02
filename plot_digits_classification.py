from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib
from utilities import preprocess, train_model, split_train_dev_test, read_digits, predict_and_eval,split_train_dev_test,tune_hyperparameters, ensure_directory_exists

x, y = read_digits()

import itertools

# Define the ranges of development and test sizes
dev_size_options = [0.1, 0.2, 0.3]
test_size_options = [0.1, 0.2, 0.3]

# Generate combinations of development and test sizes
dev_test_combinations = [{'test_size': test, 'dev_size': dev} for test, dev in itertools.product(test_size_options, dev_size_options)]

# Define model types
model_types = ['svm', 'decision_tree']

for model_type in model_types:
    for dict_size in dev_test_combinations:
        test_size_options = dict_size['test_size']
        dev_size_options = dict_size['dev_size']
        train_size = 1 - (dev_size_options + test_size_options)
        
        # Data splitting into train, test, and dev set
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=test_size_options, dev_size=dev_size_options)
        
        # Data Preprocessing
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)
        X_dev = preprocess(X_dev)
        
        # Defining the list hyper-params for SVM Classifier or Decision Tree Classifier
        if model_type == 'svm':
            gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
            C_range = [0.1, 1.0, 2, 5, 10]
            #param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in itertools.product(gamma_range, C_range)]
            param_combinations = [{'gamma': [gamma], 'C': [C]} for gamma, C in itertools.product(gamma_range, C_range)]
        elif model_type == 'decision_tree':
            max_depth_range = [None, 10, 20, 30]
            min_samples_split_range = [2, 5, 10]
            #param_combinations = [{'max_depth': max_depth, 'min_samples_split': min_samples_split} for max_depth, min_samples_split in itertools.product(max_depth_range, min_samples_split_range)]
            param_combinations = [{'max_depth': [max_depth], 'min_samples_split': [min_samples_split]} for max_depth, min_samples_split in itertools.product(max_depth_range, min_samples_split_range)]

        # Tuning the Hyperparameters
        train_acc, best_hparams, best_model, best_accuracy = tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations, model_type)
        
        # Train the data
        model = train_model(X_train, y_train, best_hparams, model_type)
        
        # Predict and evaluate the model on the test subset
        accuracy_test = predict_and_eval(model, X_test, y_test)
        
        # Print best params for each of the 9 combinations
        print(f'Model Type: {model_type}, test_size={test_size_options}, dev_size={dev_size_options}, train_size={train_size}, train_acc:{train_acc:.2f} dev_acc:{best_accuracy:.2f} test_acc: {accuracy_test:.2f}')
        print(f' Best params:{best_hparams}')


    # After selecting the best model, calculate accuracy and print confusion matrix
    best_train_pred = best_model.predict(X_test)
    best_dev_pred = best_model.predict(X_dev)

    train_accuracy = accuracy_score(y_test, best_train_pred)
    dev_accuracy = accuracy_score(y_dev, best_dev_pred)
    
    print("Final Best model parameters:", best_hparams)
    print("Training accuracy with best model:", train_accuracy)
    print("Development accuracy with best model:", dev_accuracy)
    
    # Print confusion matrices
    print("Confusion matrix for training data:")
    print(confusion_matrix(y_test, best_train_pred))
    
    print("Confusion matrix for development data:")
    print(confusion_matrix(y_dev, best_dev_pred))

    # Save the best model to a file
    if model_type == 'svm':
        best_model_filename = f"best_svm_model_{model_type}_{'_'.join([f'{k}_{v}' for k, v in best_hparams.items()])}.pkl"
    elif model_type == 'decision_tree':
        best_model_filename = f"best_decision_tree_model_{model_type}_{'_'.join([f'{k}_{v}' for k, v in best_hparams.items()])}.pkl"

    shared_volume_path = 'models'
    model_save_path = f"{shared_volume_path}/{best_model_filename}"
    ensure_directory_exists(model_save_path)


    # joblib.dump(best_model, best_model_filename)
    joblib.dump(best_model, model_save_path)
    print(f"Best {model_type} model saved as {best_model_filename}")