# small_train_dataset.py

import pandas as pd

full = pd.read_csv("../../dataset/model/train_9mer_5050.csv")

small = full.sample(n = 20)
small_test = full.sample(n = 20)

#small = full[:10]
#small_test = full[-10:]

small.to_csv("../../dataset/model/small_train_9mer.csv", index = None)
small_test.to_csv("../../dataset/model/small_test_9mer.csv", index=None)
