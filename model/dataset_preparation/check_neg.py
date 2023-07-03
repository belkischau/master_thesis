import pandas as pd 

neg_df = pd.read_csv("../../dataset/model/negative_extended_with_label_9mer.csv")
pos_df = pd.read_csv("../../dataset/model/positive_extended_with_label_9mer.csv", header=None)

neg_df = neg_df.drop(columns=['target'])

pos_df.columns = ['protein', 'peptide', 'before', 'after', 'extended', 'target']
pos_df = pos_df.drop(columns=['target'])

combined = pd.merge(neg_df, pos_df, how="inner", on=['protein', 'peptide', 'before', 'after', 'extended'])

#combined = pd.merge(neg_df, pos_df, how="inner", on=["extended"])
print(f"overlaped rows = {combined.shape[0]}")
print(combined.head())

'''


neg_set = set(neg_df['extended'])
pos_set = set(pos_df['extended'])

print(f"number of data in pos = {len(pos_set)}")
print(f"number of data in neg = {len(neg_set)}")

print(f"difference = {len(neg_set.difference(pos_set))}")
print(f"overlapped = {len(neg_set.intersection(pos_set))}")
'''
