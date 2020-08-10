import DataManager as manager
import DataPuller as puller
import Predictors as pr
import plotly.express as ex
import numpy as np

# data = manager.pull_prepare_save("1997-01-01", "2020-08-11", "1d")
# pr.train_eval_save("MLP", data, filename="test")

# PULL APPEND SAVE
# data = manager.load("1997-01-01-2020-07-19:1d")
# data = manager.pull_append_prepare_save("2020-07-20", "2020-07-31", "1d", data)
# data = manager.load("1997-01-01-2020-07-19:1d")
# data = manager.pull_prepare_save("1997-01-01", "2020-07-19", "1mo")
#---------------------------------------------------------

# LOAD APPEND
data = manager.load("1997-01-01-2020-07-31:1d")
# data1wk = manager.load("1997-01-01-2020-07-19:1wk")
# data1mo = manager.load("1997-01-01-2020-07-19:1mo")

# data = data1d.append(data1wk, ignore_index=True)
# data = data.append(data1mo, ignore_index=True)
#---------------------------------------------------------

# TRAIN EVAL SAVE
pr.train_eval_save("MLP", data, filename="MLP_8")
# pr.train_eval_save("BAYES", data, filename="BAYES_3")
# pr.train_eval_save("RF", data, filename="RF_3")
#---------------------------------------------------------

# LOAD EVAL SAVE
# pr.load_eval_save("MLP_5", data, "MLP_5")
# pr.load_eval_save("BAYES", data, "BAYES_3")
# pr.load_eval_save("RF", data, "RF_3")
#----------------------------------------------------------

# SIMPLE EVAL
# data = manager.load("2020-07-20-2020-07-31:1d")
# mlp1 = pr.mlp_load("MLP_2")
# mlp2 = pr.mlp_load("MLP_3")
# mlp3 = pr.mlp_load("MLP_4")
# # rf = pr.random_forests_load("RF_2")
# # bayes = pr.bayes_load("BAYES_2")
# pre1, _, _, std = pr.evaluate(mlp1, data[-10:], "inter-diff", True)
# pre2, _, _, std = pr.evaluate(mlp2, data[-10:], "inter-diff", True)
# pre3, _, _, std = pr.evaluate(mlp3, data[-10:], "inter-diff", True)
# # pre2, ae, mae, std = pr.evaluate(rf, data[-5:], "inter-diff", True)
# # pre3, ae, mae, std = pr.evaluate(bayes, data[-5:], "inter-diff", True)

# high = (data["high"] - data["open"])/data["open"] * 100
# low = (data["low"] - data["open"])/data["open"] * 100
# actual = data["inter-diff"]
# print("ACTUAL:")
# print(data["inter-diff"][-10:])
# print("HIGH:")
# print(high[-10:])
# print("LOW:")
# print(low[-10:])

# print("MLP1:")
# print(pre1)
# print("MLP2:")
# print(pre2)
# print("MLP3:")
# print(pre3)

# actual = actual[-9:].values
# high = high[-9:].values
# low = low[-9:].values
# actual = np.append(actual, 0)
# high = np.append(high, 0)
# low = np.append(low, 0)
# pre4 = (pre1+pre2+pre3) / 3
# fig = ex.line(   x=np.arange(10), 
#                     y=[ actual, 
#                         high,  
#                         low,
#                         pre1.astype(np.float64), 
#                         pre2.astype(np.float64), 
#                         pre3.astype(np.float64),
#                         pre4.astype(np.float64)], 
#                     labels=["actual", "high", "low", "mlp1", "mlp2", "mlp3"])
# fig.show()
# print("RF:")
# print(pre2)
# print("BAYES:")
# print(pre3)
#----------------------------------------------------------
