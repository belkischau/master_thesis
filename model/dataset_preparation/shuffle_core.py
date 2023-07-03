import pandas as pd
import numpy as np
import os

for i in range(5):
    df = pd.read_csv(f"../../dataset/model/partition/c00{i}_9mer.csv")

    before_list = []
    after_list = []
    core_list = []
    shuffle_core = []
    shuffle_extended = []

    for p in df['extended'].values.ravel():
        core = p[6:15]
        core_list.append(core)
        before = p[0:6]
        before_list.append(before)
        after = p[15:21]
        after_list.append(after)

        seed_value=np.random.randint(0,100)
        np.random.seed(seed_value)
        l = list(core)
        np.random.shuffle(l)
        new_core = ''.join(l)
        shuffle_core.append(new_core)
        new_extended = before + new_core + after
        shuffle_extended.append(new_extended)

    df['before'] = before_list
    df['core'] = core_list
    df['shuffle_core'] = shuffle_core
    df['after'] = after_list
    df['shuffle_extended'] = shuffle_extended
    print(df.head())

    df.to_csv(f'../../dataset/model/partition/c00{i}_9mer_shuffle.csv', index=False)

#    break
