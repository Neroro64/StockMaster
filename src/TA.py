import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from sklearn.decomposition import PCA
import math
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

PATH = "Data/DAX/yahoo_fin/"

def load(filename="1d"):
     data = pd.read_csv(PATH+"{}.csv".format(filename))
     return data
def save(filename, data):
     data.to_csv(PATH+filename+".csv")

def normalize(x):
     mean = np.mean(x, axis=0, keepdims=True)
     std = np.std(x, axis=0, keepdims=True)
     return (x - mean) / std
def random_forests(period):
     feature_list = ["1-rad", "inter-derivative", "bb-upper", "bb-lower", "bb-middle", "sma-cross", "macdsignal"]
     features = period[feature_list]#[:-1]
     features = features.values
     features = normalize(features)

     targets = period["inter-diff"]#[1:]
     targets = targets.values
     targets = normalize(targets)

     # Using Skicit-learn to split data into training and testing sets
     train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = 0.20, random_state = 2020)


     rf = RandomForestRegressor(n_estimators = 1000, random_state = 2020, max_depth=30, criterion="mae")# Train the model on training data
     rf.fit(train_features, train_labels)

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

     fig = px.scatter(x=predictions, y=test_labels)
     fig.show()

def bayes(period):
     feature_list = ["1-rad", "inter-derivative", "bb-upper", "bb-lower", "bb-middle", "sma-cross", "macdsignal"]
     # feature_list = ["inter-derivative", "1-rad", "bb-upper", "bb-lower", "bb-middle", "ema-cross", "sma-cross", "rsi", "stoch-diff", "sar-diff", "macdsignal","willr"]
     features = period[feature_list]#[:-1]
     features = features.values
     features = normalize(features)

     targets = period["inter-diff"]#[1:]
     targets = targets.values
     targets = normalize(targets)

     train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = 0.20, random_state = 2020)

     clf = BayesianRidge(compute_score=True)
     clf.fit(train_features, train_labels)

     # Use the forest's predict method on the test data
     predictions = clf.predict(test_features)# Calculate the absolute errors
     errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
     print('Mean Absolute Error:', round(np.mean(errors), 2))

     # Calculate mean absolute percentage error (MAPE)
     mape = 100 * (abs(errors) / abs(test_labels))# Calculate and display accuracy
     accuracy = 100 - np.mean(mape)
     print('Accuracy:', round(accuracy, 2), '%.')

     fig = px.scatter(x=predictions, y=test_labels)
     fig.show()

data = load("day") 
period = data[200:1200]
bayes(period)
# random_forests(period)
# feature_list = ["inter-derivative", "1-rad", "bb-upper", "bb-lower", "bb-middle", "ema-cross", "sma-cross", "rsi", "stoch-diff", "sar-diff", "macdsignal", "3-rad", "5-rad", "20-rad","willr"]
