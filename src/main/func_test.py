import DataManager as manager
import DataPuller as puller
import Predictors as pr

# data = manager.pull_prepare_save("1997-01-01", "2020-07-19", "wk")
# pr.train_eval_save("MLP", data, filename="test")

# data = manager.pull_append_prepare_save("2011-01-01", "2012-01-01", "1d", data)
# print(data)
# data = manager.load("1997-01-01-2020-07-19:1d")
# data = manager.pull_prepare_save("1997-01-01", "2020-07-19", "1mo")

data1d = manager.load("1997-01-01-2020-07-19:1d")
data1wk = manager.load("1997-01-01-2020-07-19:1wk")
data1mo = manager.load("1997-01-01-2020-07-19:1mo")

data = data1d.append(data1wk, ignore_index=True)
data = data.append(data1mo, ignore_index=True)

pr.train_eval_save("MLP", data, filename="MLP_2")
pr.train_eval_save("BAYES", data, filename="BAYES_2")
pr.train_eval_save("RF", data, filename="RF_2")
