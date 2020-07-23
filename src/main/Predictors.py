import numpy as np
import pandas as pd
from scipy import stats
import math
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
import plotly.express as px
import DataManager

PATH = "/home/nuoc/Documents/StockMaster/src/models/"
RF_FEATURES = ["inter-derivative", "1-rad", "bb-upper", "bb-lower", "bb-middle"]
BAYES_FEATURES = ["inter-derivative", "1-rad", "bb-upper", "bb-lower", "bb-middle", "ema-cross", "rsi", "stoch-diff", "sar-diff", "macdsignal"]
MLP_FEATURES = ["1-rad", "inter-derivative", 
                    "bb-upper", "bb-lower", "bb-middle", 
                    "ema-cross", "macdsignal", "macdhist", "macd", 
                    "rsi", "sar-diff", "stoch-diff", 
                    "intra-derivative", "1-err", "intra-diff"]


def normalize(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / std

def random_forests_train(data, test_size=0.2, filename=None, N=1000, max_depth=30, seed=2020, verbose=True):
    feature_list = RF_FEATURES
    features = data[feature_list]#[:-1]
    features = np.nan_to_num(features.values[:-1])
    features = normalize(features)

    targets = data["inter-diff"]#[1:]
    targets = targets.values[1:]

    # Using Skicit-learn to split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = test_size, random_state = seed)


    rf = RandomForestRegressor(n_estimators = N, random_state = seed, max_depth=max_depth, criterion="mae")# Train the model on training data
    rf.fit(train_features, train_labels)

    if verbose:
        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)# Calculate the absolute errors
        errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2))

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (abs(errors) / abs(test_labels))# Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')

        # Get numerical feature importances
        importances = list(rf.feature_importances_)# List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

        p = predictions > 0
        t = test_labels > 0
        signs = p == t
        print("Direction accuracy: ", (np.sum(signs) / len(signs))*100, "%")

        fig = px.scatter(x=predictions, y=test_labels)
        fig.show()

    if not filename == None:
        with open(PATH+filename+".model", 'wb') as f:
            pickle.dump(rf, f)
    return rf

def random_forests_load(filename):
    with open(PATH+filename+".model", 'rb') as f:
        rf = pickle.load(f)
    return (rf, RF_FEATURES)


def bayes_train(data, test_size=0.2, filename=None, seed=2020, verbose=True):
    feature_list = BAYES_FEATURES
    features = data[feature_list]#[:-1]
    features = np.nan_to_num(features.values[:-1])
    features = normalize(features)

    targets = data["inter-diff"]#[1:]
    targets = targets.values[1:]

    train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = test_size, random_state = 2020)

    clf = BayesianRidge(compute_score=True)
    clf.fit(train_features, train_labels)

    if verbose:
        # Use the forest's predict method on the test data
        predictions = clf.predict(test_features)# Calculate the absolute errors
        errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2))

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (abs(errors) / abs(test_labels))# Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')

        p = predictions > 0
        t = test_labels > 0
        signs = p == t
        print("Direction accuracy: ", (np.sum(signs) / len(signs))*100, "%")

        fig = px.scatter(x=predictions, y=test_labels)
        fig.show()
    if not filename == None:
        with open(PATH+filename+".model", 'wb') as f:
            pickle.dump(clf, f)
    return clf

def bayes_load(filename):
    with open(PATH+filename+".model", 'rb') as f:
        bayes = pickle.load(f)
    return (bayes,BAYES_FEATURES)

def mlp_train(data, test_size=0.2,  filename=None, batch_size=100, epochs=800):
    feature_list = MLP_FEATURES
    N = len(feature_list)
    features = data[feature_list]
    features = np.nan_to_num(features.values[:-1])
    features = normalize(features)

    targets = data["inter-diff"][1:]
    targets = targets.values

    train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = test_size, random_state = 2020)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(N),
        tf.keras.layers.Dense(4*N, activation='relu'),
        tf.keras.layers.Dense(2*N, activation='relu'),
        tf.keras.layers.Dense(N, activation='relu'),
        tf.keras.layers.Dense(N/2, activation='relu'),
        tf.keras.layers.Dense(1)
        ]) 

    model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])


    model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size)
    model.evaluate(test_features,  test_labels, verbose=2)

    if not filename == None:
        model.save(PATH+filename+".h5")

    return model

def mlp_load(filename):
    model = tf.keras.models.load_model(PATH+filename+".h5")
    return (model, MLP_FEATURES)


def predict(predictor, data):
    model = predictor[0]
    feature_list = predictor[1]

    features = data[feature_list].values
    predictions = model.predict(features)
    return predictions

def evaluate(predictor, data, target="inter-diff", all=False):
    model = predictor[0]
    feature_list = predictor[1]

    if all:
        features = data[feature_list].values
        labels = data[target].values
    else:
        features = data[feature_list].values[:-1]
        labels = data[target][1:].values

    features = np.nan_to_num(features)
    features = normalize(features)

    predictions = model.predict(features).ravel()
    ae = np.abs(labels - predictions)
    mae = np.mean(ae)
    std = np.std(ae)

    return predictions, ae, mae, std

def compile_result(name, predictor, data, target="inter-diff", file=None):

    predictions, ae, mae, std = evaluate(predictor, data, target)
    results = pd.DataFrame(data={
        name+"_actual" : data[target][1:],
        name+"_predicted" : predictions.ravel(),
        name+"_AE" : ae.ravel(),
        name+"_MAE" : mae,
        name+"_SPREAD" : std
    })

    if not file == None:
        result_file = DataManager.load(file)
        result_file.append(results, ignore_index=True)
        DataManager.save(file, result_file)
    else:
        DataManager.save(name+"_results", results)

    return predictions, mae, std

def train_eval_save(name, data, test_size=0.1, filename=None, log=True):
    if name == "RF":
        model = random_forests_train(data, test_size, filename=filename, N=1000, max_depth=30, seed=2020, verbose=True)
        feature_list = RF_FEATURES
    elif name == "BAYES":
        model = bayes_train(data, test_size, filename=filename, seed=2020, verbose=True)
        feature_list = BAYES_FEATURES
    elif name == "MLP":
        model = mlp_train(data, test_size, filename=filename, batch_size=100, epochs=1200)
        feature_list = MLP_FEATURES
    else:
        return -1
    
    predictions, mae, std = compile_result(name, (model, feature_list), data, "inter-diff")
    # if log:
    #     print("Training results: ")
    #     print(predictions)
    #     print(mae)
    #     print(std)
    #     print("-"*70)
    
def load_eval_save(name, data, filename=None):
    if name == "RF":
        predictor = random_forests_load("RF_2")
    elif name == "BAYES":
        predictor = bayes_load("BAYES_2")
    elif name == "MLP":
        predictor = mlp_load("MLP_2")
    else:
        return -1

    predictions, mae, std = compile_result(name, predictor, data, "inter-diff")












    



        
    