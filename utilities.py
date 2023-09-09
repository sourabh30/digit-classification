from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# Function for splitting the data set inot train, test and dev set
def split_train_dev_test(X, y, test_size=0.2, dev_size=0.25, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=dev_size / (dev_size + test_size), random_state=random_state
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


# Function for evaluationg the model
def predict_and_eval(model, X, y):
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    # classification_rep = classification_report(y, y_pred)
    # confusion_mat = confusion_matrix(y, y_pred)
    # return accuracy, classification_rep, confusion_mat
    return accuracy

def tune_hyperparameters(X_train, y_train, X_dev, y_dev, hyperparameter_combinations):
    best_hyperparameters = None
    best_model = None
    best_dev_accuracy = 0.0
    
    # Iterate through each set of hyperparameters in the list
    for hyperparameters in hyperparameter_combinations:
        # Train a model with the current set of hyperparameters
        current_model = train_model(X_train, y_train, hyperparameters)

        # Evaluate the model's accuracy on the training dataset
        train_accuracy = predict_and_eval(current_model, X_train, y_train)  
        
        # Evaluate the model on the development dataset
        dev_accuracy = predict_and_eval(current_model, X_dev, y_dev)  
        
        # Check if this model's accuracy is better than the current best
        if dev_accuracy > best_dev_accuracy:
            best_hyperparameters = hyperparameters
            best_model = current_model
            best_dev_accuracy = dev_accuracy
    
    return train_accuracy, best_hyperparameters, best_model, best_dev_accuracy

# Train a specified model on the given data
def train_model(X, y, model_params, model_type='svm'):
    if model_type == 'svm':
        classifier = svm.SVC(**model_params)
    classifier.fit(X, y)
    return classifier


def preprocess(x):
    num_samples = len(x)
    x = x.reshape((num_samples, -1))
    return x