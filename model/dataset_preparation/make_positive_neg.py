import pandas as pd
import os 
import sys 

pos_df = pd.read_csv('../dataset/included_peptides_unique_pairs.csv')

neg_df1 = pd.read_csv('../dataset/not_included_peptides_unique_pairs.csv')
neg_df2 = pd.read_csv('../dataset/iedb_not_included_peptides_unique_pairs.csv')

combined_neg = pd.concat([neg_df1, neg_df2])

pos_df.to_csv('../dataset/model/positive.csv', index=None)
combined_neg.to_csv('../dataset/model/negative.csv', index=None)

pos_df['target'] = 1

neg_df1['target'] = 0
neg_df2['target'] = 0

combined_neg_with_label = pd.concat([neg_df1, neg_df2])

pos_df.to_csv('../dataset/model/positive_with_label.csv', index=None)
combined_neg_with_label.to_csv('../dataset/model/negative_with_label.csv', index=None)
