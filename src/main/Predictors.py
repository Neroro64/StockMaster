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

def normalize(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / std

def random_forests_train(data, filename=None, N=1000, max_depth=30, seed=2020, verbose=True):
    feature_list = ["inter-derivative", "1-rad", "bb-upper", "bb-lower", "bb-middle"]
    features = data[feature_list]#[:-1]
    features = features.values[:-1]

    targets = data["inter-diff"]#[1:]
    targets = targets.values[1:]

    # Using Skicit-learn to split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = 0.20, random_state = seed)


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
        with open(filename, 'wb') as f:
            pickle.dump(rf, f)
    return rf

def bayes_train(data, filename=None, seed=2020, verbose=True):
     feature_list = ["inter-derivative", "1-rad", "bb-upper", "bb-lower", "bb-middle", "ema-cross", "rsi", "stoch-diff", "sar-diff", "macdsignal"]
     features = data[feature_list]#[:-1]
     features = features.values[:-1]

     targets = data["inter-diff"]#[1:]
     targets = targets.values[1:]

     train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = 0.20, random_state = 2020)

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
        with open(filename, 'wb') as f:
            pickle.dump(clf, f)
    return clf

def mlp_train(data, batch_size=100, epochs=600):
    feature_list = ["1-rad", "inter-derivative", 
                    "bb-upper", "bb-lower", "bb-middle", 
                    "ema-cross", "macdsignal", "macdhist", "macd", 
                    "rsi", "sar-diff", "stoch-diff", 
                    "intra-derivative", "1-err", "intra-diff"]

    N = len(feature_list)
    features = data[feature_list][:-1]
    features = features.values
    features = normalize(features)

    targets = data["inter-diff"][1:]
    targets = targets.values

    train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = 0.20, random_state = 2020)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(N),
        tf.keras.layers.Dense(2*N, activation='relu'),
        tf.keras.layers.Dense(N, activation='relu'),
        tf.keras.layers.Dense(1)
        ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])


    model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size)
    model.evaluate(test_features,  test_labels, verbose=2)
    

        
    