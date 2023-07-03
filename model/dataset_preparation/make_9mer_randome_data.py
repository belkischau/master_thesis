import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = '/home/projects/vaccine/people/belcha/master_thesis/dataset/model'

pos_df = pd.read_csv(f'{dataset_path}/positive_extended_with_label.csv')
neg_df = pd.read_csv(f'{dataset_path}/negative_extended_with_label.csv')

pos_df = pos_df[(pos_df['peptide'].str.len() == 9) & (pos_df['extended'].str.len() == 21)]
neg_df = neg_df[(neg_df['peptide'].str.len() == 9) & (neg_df['extended'].str.len() == 21)]

print(f"positive: no. of rows before removing uncommon AA = {pos_df.shape[0]}")
print(f"negative: no. of rows before removing uncommon AA = {neg_df.shape[0]}")

pos_df = pos_df[~pos_df['extended'].str.contains('U|J|X')]
neg_df = neg_df[~neg_df['extended'].str.contains('U|J|X')]

print(f'Number of rows in positive 9mer data: {pos_df.shape[0]}. ')
print(f'Number of rows in negative 9mer data: {neg_df.shape[0]}. \n')

pos_df.to_csv(f'{dataset_path}/positive_extended_with_label_9mer.csv', index=None)
neg_df.to_csv(f'{dataset_path}/negative_extended_with_label_9mer.csv', index=None)

# ====== 1:1 ===========

df = pd.concat([pos_df, neg_df])
df = df.sample(frac = 1)

train_df, test_df = train_test_split(df, test_size=0.2)

print("positve : negative = 1:1")
print("Training dataframe: ", len(train_df), "rows")
print("Testing dataframe: ", len(test_df), "rows\n")

train_df.to_csv(f'{dataset_path}/train_9mer_all.csv', index=None)
test_df.to_csv(f'{dataset_path}/test_9mer_all.csv', index=None)


# ====== 1:5 ============

neg_random_rows = neg_df.sample(n = pos_df.shape[0])
neg_5050 = neg_random_rows.loc[neg_random_rows.index]

df_5050 = pd.concat([pos_df, neg_5050])
df_5050 = df_5050.sample(frac = 1)

train_5050, test_5050 = train_test_split(df_5050, test_size=0.2)
print("positve : negative = 1:5")
print("Training dataframe: ", len(train_5050), "rows")
print("Testing dataframe: ", len(test_5050), "rows\n")

train_5050.to_csv(f'{dataset_path}/train_9mer_5050.csv', index=None)
test_5050.to_csv(f'{dataset_path}/test_9mer_5050.csv', index=None)

# ======= 1:10 =============

a = pos_df.shape[0] * 10

neg_random_rows_10 = neg_df.sample(n = a)
neg_1090 = neg_random_rows_10.loc[neg_random_rows_10.index]

df_1090 = pd.concat([pos_df, neg_1090])
df_1090 = df_1090.sample(frac = 1)

train_1090, test_1090 = train_test_split(df_1090, test_size=0.2)
print("positve : negative = 1:10")
print("Training dataframe: ", len(train_1090), "rows")
print("Testing dataframe: ", len(test_1090), "rows")

train_1090.to_csv(f'{dataset_path}/train_9mer_1090.csv', index=None)
test_1090.to_csv(f'{dataset_path}/test_9mer_1090.csv', index=None)
