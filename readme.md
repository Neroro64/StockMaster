# Stock Master
A web application project that aims to display stock information, technical indicators, to predict future trend, momentum, volumn and volatility using machine learning / deep learning, and to form a strategy based on the predictions to maximize the potential profit.

## A brief description
The project consists of 3 parts. Web app - Predictor - Strategist

### Web app
An accessible application that displays the graph of index growth during variable periods and intervals. The data will be pulled from Yahoo Finance using [yahoo-fin](https://pypi.org/project/yahoo-fin/). It should also be able to calculate and display the following technical indicators:
#### Trend
    1. EMA cross
    2. MACD
    3. Bollinger bands
    4. Parabolic Stop and Reverse (SAR)
    5. Average Directional Index (ADX)

#### Momentum
    1. Ichimoku Cloud
    2. Relative Strength Index (RSI)
    3. Stochastic

### Predictor
A ML/DL model that uses the indicators and the prices to predict future movement.
#### Models to experiement:
    1. Random regression forests
    2. Probabilistic modeling
    3. MLP
    4. ConvNet
#### Outputs:
    The model should output predictions about the very next period(s)
    1. Trend
    2. Momentum
    3. Derivative of price change

#### Libraries:
    Tensorflow
    Scipy

### Strategist
A program that takes outputs from the Predictor and determines the best (most-profitable) action policy.
The idea is that there are a number of presets of parameters (risk acceptance, profit/loss ratios etc), and the program form policies based on the presets, and ran simulations of historical data (with predictions) to test each policy. 

With other words, given the predictions (% up, % down), the program returns a table of max profit and its probability, max loss and its probability and determine whether go bull or bear based on the presets.


