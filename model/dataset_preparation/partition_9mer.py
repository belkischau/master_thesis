import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import numpy as np 

#path = '../hobohm/splitting'
path = '../../dataset/model/part'

#files = ['c000', 'c001', 'c002', 'c003', 'c004', 'f000', 'f001', 'f002', 'f003', 'f004']
files = ['c000', 'c001', 'c002', 'c003', 'c004']

save_path = '../../dataset/model/partition'

if not os.path.exists(save_path):
    os.makedirs(save_path)


#neg_df = pd.read_csv("../../dataset/model/negative_extended_9mer_with_label.csv")
neg_df = pd.read_csv("../../dataset/model/negative_extended_with_label_9mer.csv")

neg_df = neg_df[['protein', 'peptide', 'before', 'after', 'extended', 'target']].drop_duplicates()
neg_df = neg_df[['extended', 'target']]
#neg_df = neg_df.sample(n=94465, random_state=1)      # 94465 = total number of peptides in c000+f000
neg_df = neg_df.sample(n=138773, random_state=1)      # 138773 = total number of peptides in all c00X
neg_df_shuffled = neg_df.sample(frac=1) 



'''
neg_df_splits = np.array_split(neg_df_shuffled, 5)  

n000 = neg_df_splits[0]
n001 = neg_df_splits[1]
n002 = neg_df_splits[2]
n003 = neg_df_splits[3]
n004 = neg_df_splits[4]

n_list = [n000, n001, n002, n003, n004]

nf000 = pd.concat([n001, n002, n003, n004])
nf001 = pd.concat([n000, n002, n003, n004])
nf002 = pd.concat([n000, n001, n003, n004])
nf003 = pd.concat([n000, n001, n002, n004])
nf004 = pd.concat([n000, n001, n002, n003])

nf_list = [nf000, nf001, nf002, nf003, nf004]
'''

print(f"Number of data in raw neg df: {neg_df.shape[0]} ")

for file in files:
    print(f"\nfile = {file}")
    pos_df = pd.read_csv(f'{path}/{file}', header=None, sep=' ')
    pos_df.rename(columns = {0:'extended'}, inplace = True)
#    df = df[df.iloc[:,0].apply(lambda x: len(x) == 21)]
#    df = df[~df['extended'].str.contains('U|J|X')]
    print(f"before droping duplicates: {pos_df.shape[0]}")
    print(f"unique value in pos df: {len(pos_df['extended'].unique())}")
    pos_df = pos_df[['extended']].drop_duplicates()
    pos_df['target'] = 1
#    pos_df = pos_df[['extended', 'target']].drop_duplicates()

    selected_neg_df = neg_df_shuffled.sample(n=pos_df.shape[0], random_state=24)
    df = pd.concat([pos_df, selected_neg_df])
    df = df.sample(frac = 1)
 
    print(df.head())
    print(f"Number of data in pos df: {pos_df.shape[0]}")
    print(f"Number of data in neg df: {selected_neg_df.shape[0]}")
    print(f"Number of data in concat: {df.shape[0]}")
    print(f"Number of pos in concat df: {len(df[df['target']==1])}")
    print(f"Number of neg in concat df: {len(df[df['target']==0])}")

    neg_df_shuffled = neg_df_shuffled.drop(selected_neg_df.index[:])
    print(f"After removing the random row, number of data in neg df: {neg_df_shuffled.shape[0]}\n")

    df.to_csv(f'{save_path}/{file}_9mer.csv', index=None, sep=',')

'''












#    df.to_csv(f'{save_path}/{file}_9mer.csv', index=None, sep=',')

#    negative_df = pd.read_csv('../../dataset/model/negative_extended_with_label_9mer.csv')

#    negative_unique = negative_df[['protein', 'peptide', 'before', 'after', 'extended', 'target']].drop_duplicates()

#    negative_random_rows = negative_unique.sample(n=65050)
#    negative_random = negative_random_rows.loc[negative_random_rows.index]
#    negative_random = negative_random[['extended', 'target']]
#    print(negative_random.head())    

#    neg_train, neg_test = train_test_split(negative_random, test_size=0.25)


    if file.startswith('c'): 
        neg = n_list[int(file[-1])]
#        print(f"pos file = \n{df.head()}")
#        print(f"neg file to be attached = \n{neg.head()}")
        test_df = pd.concat([df, neg])
        test_df = test_df.sample(frac = 1)
#        print(test_df.head())
#        print(f"\nNumber of positive and negative in {file}:\n{test_df['target'].value_counts()}")
#        break
        test_df.to_csv(f'{save_path}/{file}_9mer.csv', index=None, sep=',')
        
    elif file.startswith('f'):
        neg = nf_list[int(file[-1])]
#        print(f"pos file = \n{df.head()}")
#        print(f"neg file to be attached = \n{neg.head()}")
        train_df = pd.concat([df, neg])
        train_df = train_df.sample(frac = 1)
#        print(test_df.head())
#        print(f"\nNumber of positive and negative in {file}:\n{train_df['target'].value_counts()}")
#        break
        train_df.to_csv(f'{save_path}/{file}_9mer.csv', index=None, sep=',')


'''
