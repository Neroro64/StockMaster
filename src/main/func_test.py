import DataManager as manager
import DataPuller as puller
import Predictors as pr
import plotly.express as ex

# data = manager.pull_prepare_save("1997-01-01", "2020-08-06", "1d")
# pr.train_eval_save("MLP", data, filename="test")

# PULL APPEND SAVE
# data = manager.load("1997-01-01-2020-08-05:1d")
# data = manager.pull_append_prepare_save("2020-07-20", "2020-07-31", "1d", data)
# data = manager.load("1997-01-01-2020-07-19:1d")
# data = manager.pull_prepare_save("1997-01-01", "2020-07-19", "1mo")
#---------------------------------------------------------

# LOAD APPEND
data = manager.load("1997-01-01-2020-08-06:1d")
# data1wk = manager.load("1997-01-01-2020-07-19:1wk")
# data1mo = manager.load("1997-01-01-2020-07-19:1mo")

# data = data1d.append(data1wk, ignore_index=True)
# data = data.append(data1mo, ignore_index=True)
#---------------------------------------------------------

# TRAIN EVAL SAVE
# pr.train_eval_save("MLP", data, filename="MLP_4")
# pr.train_eval_save("BAYES", data, filename="BAYES_3")
# pr.train_eval_save("RF", data, filename="RF_3")
#---------------------------------------------------------

# LOAD EVAL SAVE
# pr.load_eval_save("MLP", data, "MLP_4")
# pr.load_eval_save("BAYES", data, "BAYES_3")
# pr.load_eval_save("RF", data, "RF_3")
#----------------------------------------------------------

# SIMPLE EVAL
# data = manager.load("2020-07-20-2020-07-31:1d")
mlp = pr.mlp_load("MLP_4")
# rf = pr.random_forests_load("RF_2")
# bayes = pr.bayes_load("BAYES_2")
pre1, ae, mae, std = pr.evaluate(mlp, data[-5:], "inter-diff", True)
# pre2, ae, mae, std = pr.evaluate(rf, data[-5:], "inter-diff", True)
# pre3, ae, mae, std = pr.evaluate(bayes, data[-5:], "inter-diff", True)

high = (data["high"] - data["open"])/data["open"] * 100
low = (data["low"] - data["open"])/data["open"] * 100

print("ACTUAL:")
print(data["inter-diff"][-5:])
print("HIGH:")
print(high[-5:])
print("LOW:")
print(low[-5:])

print("MLP:")
print(pre1)
print("RF:")
print(pre2)
print("BAYES:")
print(pre3)
#----------------------------------------------------------
