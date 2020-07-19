import DataManager as manager
import DataPuller as puller
import Predictors as pr

data = manager.pull_prepare_save("1997-01-01", "2020-07-19", "1d")
# pr.train_eval_save("MLP", data, filename="test")

# data = manager.pull_append_prepare_save("2011-01-01", "2012-01-01", "1d", data)
# print(data)
pr.train_eval_save("MLP", data, filename="MLP_1")