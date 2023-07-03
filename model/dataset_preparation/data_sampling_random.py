import pandas as pd
from sklearn.model_selection import train_test_split

negative = pd.read_csv('../../dataset/model/negative_extended_9mer_with_label.csv')

positive = pd.read_csv('../../dataset/model/positive_extended_9mer_with_label.csv')

negative_unique = negative[['protein', 'peptide', 'before', 'after', 'extended', 'target']].drop_duplicates()
positive_unique = positive[['protein', 'peptide', 'before', 'after', 'extended', 'target']].drop_duplicates()

#negative_unique.to_csv('../../dataset/model/negative_unique_extended_9mer_with_label.csv', index=False)
#positive_unique.to_csv('../../dataset/model/positive_unique_extended_9mer_with_label.csv', index=False)

# randomly sample 100 rows
#positive_random_rows = positive_unique.sample(n=10000)
#negative_random_rows = negative_unique.sample(n=positive.shape[0])

#positive_random = positive_random_rows.loc[positive_random_rows.index]
#negative_random = negative_random_rows.loc[negative_random_rows.index]
negative_random = negative_unique.sample(n=positive_unique.shape[0], random_state = 24)

#df = pd.concat([positive_random, negative_random])
df = pd.concat([positive_unique, negative_random])

df['contig'] = df['before'].str.cat(df['after'])

# split into training and testing dataframes
train_df, test_df = train_test_split(df, test_size=0.2, random_state = 14)

# print the number of rows in each dataframe
print("Training dataframe: ", len(train_df), "rows")
print("Testing dataframe: ", len(test_df), "rows")

print(f"Number of positive and negative in train dataset: \n{train_df['target'].value_counts()}")
print(f"Number of positive and negative in test dataset: \n{test_df['target'].value_counts()}")

#train_df.to_csv('../../dataset/model/train_9mer.csv', index=False)
#test_df.to_csv('../../dataset/model/test_9mer.csv', index=False)
train_df.to_csv('../../dataset/model/train_9mer_5050_contig.csv', index=False)
test_df.to_csv('../../dataset/model/test_9mer_5050_contig.csv', index=False)
