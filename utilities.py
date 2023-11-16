from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    
    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create the directory
        os.makedirs(directory)
        # print(f"Directory '{directory}' created.")
    else:
        # print(f"Directory '{directory}' already exists.")
        pass



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

def tune_hyperparameters(X_train, y_train, X_dev, y_dev, hyperparameter_combinations, model_type):
    best_hyperparameters = None
    best_model = None
    best_dev_accuracy = 0.0
    
    # Iterate through each set of hyperparameters in the list
def tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations, model_type):
    if model_type == 'svm':
        model = SVC()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    
    grid_search = GridSearchCV(model, param_combinations, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_hparams = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    
    return grid_search.best_score_, best_hparams, best_model,best_accuracy

# Train a specified model on the given data
def train_model(X_train, y_train, best_hparams, model_type='svm'):
    if model_type == 'svm':
        model = SVC(**best_hparams)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(**best_hparams)
    
    model.fit(X_train, y_train)
    return model

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 


def preprocess(x):
    num_samples = len(x)
    x = x.reshape((num_samples, -1))
    return x


def create_hparam_combo(gamma_range, C_range):
    return [{'gamma': gamma, 'C': C} for gamma in gamma_range for C in C_range]
