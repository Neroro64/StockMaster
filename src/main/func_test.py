import DataManager as manager
import DataPuller as puller
import Predictors as pr

# data = manager.pull_prepare_save("1997-01-01", "2020-07-19", "wk")
# pr.train_eval_save("MLP", data, filename="test")

# PULL APPEND SAVE
# data = manager.load("1997-01-01-2020-07-19:1d")
# data = manager.pull_append_prepare_save("2020-07-20", "2020-07-23", "1d", data)
# data = manager.load("1997-01-01-2020-07-19:1d")
# data = manager.pull_prepare_save("1997-01-01", "2020-07-19", "1mo")
#---------------------------------------------------------

# LOAD APPEND
# data1d = manager.load("1997-01-01-2020-07-19:1d")
# data1wk = manager.load("1997-01-01-2020-07-19:1wk")
# data1mo = manager.load("1997-01-01-2020-07-19:1mo")

# data = data1d.append(data1wk, ignore_index=True)
# data = data.append(data1mo, ignore_index=True)
#---------------------------------------------------------

# TRAIN EVAL SAVE
# pr.train_eval_save("MLP", data, filename="MLP_2")
# pr.train_eval_save("BAYES", data, filename="BAYES_2")
# pr.train_eval_save("RF", data, filename="RF_2")
#---------------------------------------------------------

# LOAD EVAL SAVE
# pr.load_eval_save("MLP", data, "MLP_2")
# pr.load_eval_save("BAYES", data, "BAYES_2")
# pr.load_eval_save("RF", data, "RF_2")
#----------------------------------------------------------

data = manager.load("2020-07-20-2020-07-23:1d")
mlp = pr.mlp_load("MLP_2")
rf = pr.random_forests_load("RF_2")
bayes = pr.bayes_load("BAYES_2")
pre1, ae, mae, std = pr.evaluate(mlp, data[-5:], "inter-diff", True)
pre2, ae, mae, std = pr.evaluate(rf, data[-5:], "inter-diff", True)
pre3, ae, mae, std = pr.evaluate(bayes, data[-5:], "inter-diff", True)
print(pre1)
print(pre2)
print(pre3)
print(data["inter-diff"][-4:])
