import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as ex
import talib
from scipy import stats
import math
from DataPuller import pull_dax

PATH = "../Data/DAX/yahoo_fin/"

def load(filename="1d"):
     data = pd.read_csv(PATH+"{}.csv".format(filename))
     return data
def save(filename, data):
     data.to_csv(PATH+filename+".csv")
    
def clean(data):
    id1 = 0
    for i in range(len(data["close"])):
        if (not np.isnan(data["close"][i]) and data["close"][i] != data["open"][i]):
            id1 = i
            break

    if (id1 > 0):
        samples = data.drop(np.arange(id1))
    else:
        samples = data
    
    samples = samples.drop(columns=[samples.columns[0], "ticker", "adjclose"])

    samples.reset_index(drop=True, inplace=True)
    samples = samples.drop(np.where(np.isnan(samples["close"]))[0])

    return samples

def compute_diff(data, intra=False):
    if not intra:
        close = data["close"].values
        N = len(close)
        diff = np.zeros((N, 1))
        for i in range(1, N):
            diff[i] = (close[i] - close[i-1]) / close[i-1] * 100
        return diff 

    else:
        op = data["open"].values
        cl = data["close"].values
        N = len(cl)
        diff = np.zeros(N)
        for i in range(0, N):
            diff[i] = (cl[i] - op[i]) / op[i] * 100
        return diff 

def compute_derivative(data, intra=False, h=1):
    if not intra:
        pr = data["close"]
        g = np.gradient(pr.values)
        return g
    else:
        val = data.values
        g = np.zeros(len(val))
        for i in range(len(val)):
            g[i] = np.average(np.gradient(val[i][0:4]))
        return g

def compute_direction(data, period=1):
    val = data.values
    N = len(val)
    rad = np.zeros(N)
    errors = np.zeros(N)
    arr = np.zeros(period*4)

    for i in range(N):
        arr = val[i:i+period][:,0:4].ravel()
        slope, _, _, _, std_err = stats.linregress(np.arange(len(arr)),arr)
        rad[i] = math.atan(slope)
        errors[i] = math.atan(std_err)
    return rad, errors

def prepare(raw, clean=True):
    if clean:
        data = clean(raw)
    else:
        data = raw

    intra_diff = compute_diff(data, intra=True)
    inter_diff = compute_diff(data, intra=False)
    intra_derivative = compute_derivative(data, intra=True)
    inter_derivative = compute_derivative(data, intra=False)
    rad_1d, err_1d = compute_direction(data, period=1)
    
    data["intra-diff"] = intra_diff
    data["inter-diff"] = inter_diff
    data["intra-derivative"] = intra_derivative
    data["inter-derivative"] = inter_derivative
    data["1-rad"] = rad_1d
    data["1-err"] = err_1d

    # Bollinger bands
    upper, middle, lower = talib.BBANDS(data["close"])
    data["BB-upper"] = upper
    data["BB-middle"] = middle
    data["BB-lower"] = lower
    data["bb-upper"] = data["BB-upper"] - data["close"]
    data["bb-middle"] = data["close"] - data["BB-middle"]
    data["bb-lower"] = data["close"] - data["BB-lower"]

    # EMA
    ema5 = talib.EMA(data["close"], 5)
    ema20 = talib.EMA(data["close"], 20)
    data["ema5"] = ema5
    data["ema20"] = ema20
    data["ema-cross"] = ema5 - ema20
    
    # SMA
    sma50 = talib.SMA(data["close"], 50)
    sma200 = talib.SMA(data["close"], 200)
    data["sma50"] = sma50
    data["sma200"] = sma200
    data["sma-cross"] = sma50 - sma200

    # Parabolic SAR
    sar = talib.SAR(data["high"], data["low"], 0.02, 0.2)
    data["sar"] = sar
    data["sar-diff"] = data["close"] - sar

    # ADX
    adx = talib.ADX(data["high"], data["low"], data["close"], 14)
    data["adx"] = adx
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(data["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    data["macd"] = macd
    data["macdsignal"] = macdsignal
    data["macdhist"] = macdhist

    # Momentum
    mom = talib.MOM(data["close"], timeperiod=10)
    data["mom"] = mom

    # RSI
    rsi = talib.RSI(data["close"], timeperiod=14)
    data["rsi"] = rsi

    # Stochastic 
    stoch_k, stoch_d = talib.STOCH(data["high"], data["low"], data["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    data["stoch_k"] = stoch_k
    data["stoch_d"] = stoch_d
    data["stoch-diff"] = stoch_k - stoch_d

    # Willr
    willr = talib.WILLR(data["high"], data["low"], data["close"], timeperiod=14)
    data["willr"] = willr

    return data

def pull_prepare_save(start, end, interval):
    raw = pull_dax(start, end, interval)
    data = prepare(raw)
    save("{}-{}:{}".format(start,end, interval), data)

def pull_append_prepare_save(start, end, interval, data):
    raw = pull_dax(start, end, interval)

    new_data = data[-200:].append(raw[["open", "close", "high", "low"]], ignore_index=True)
    data = data[:-200].append(prepare(new_data, False))
    return data






    
