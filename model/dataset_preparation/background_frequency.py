import pandas as pd
import os

pos_df = pd.read_csv("../../dataset/model/positive_extended_with_label_9mer.csv", header = None)
pos_df.columns = ['protein','peptide','before','after','extended','target']

neg_df = pd.read_csv("../../dataset/model/negative_extended_with_label_9mer.csv")

df0 = pd.read_csv("../../dataset/model/partition/c000_9mer.csv")
df1 = pd.read_csv("../../dataset/model/partition/c001_9mer.csv")
df2 = pd.read_csv("../../dataset/model/partition/c002_9mer.csv")
df3 = pd.read_csv("../../dataset/model/partition/c003_9mer.csv")
df4 = pd.read_csv("../../dataset/model/partition/c004_9mer.csv")

df = pd.concat([df0, df1, df2, df3, df4])
df = df[df['target'] == 0]

#print(f"number of negative from c00x = {df.shape[0]}")
#print(f"unique number of extended peptides before merging = {len(df['extended'].unique())}")

df = df.merge(neg_df, how = 'left', on = ['extended','target'])

#print(f"after merging, number of negtaive in df = {df.shape[0]}")
#print(df.head())
#print(f"unique number of extended peptides in merged df = {len(df['extended'].unique())}")

path = "../../data/fastas"

fasta_files = os.listdir(path)


aa_dict = {'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0,
           'G':0, 'H':0, 'I':0, 'J':0, 'K':0, 'L':0,
           'M':0, 'N':0, 'P':0, 'Q':0, 'R':0, 'S':0,
           'T':0, 'U':0, 'V':0, 'W':0, 'X':0, 'Y':0, 'Z':0}

proteins = df['protein'].unique()

for i in range(len(proteins)):
    protein = proteins[i]
    if protein == "H0YN83":
        protein = 'H0YMF3'
    if protein == "P0DTL6":
        protein = "E9PJD7"
    if protein == "Q5XKL5": 
        protein = "Q9UPP5"

    with open(f"{path}/{protein}.fasta", 'r') as f:
        for line in f:
            line = line.strip()
            for a in range(len(line)):
                aa_dict[line[a]] += 1
#    break

print(aa_dict)

aa_freq = {'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0,
           'G':0, 'H':0, 'I':0, 'J':0, 'K':0, 'L':0,
           'M':0, 'N':0, 'P':0, 'Q':0, 'R':0, 'S':0,
           'T':0, 'U':0, 'V':0, 'W':0, 'X':0, 'Y':0, 'Z':0}

total = sum(aa_dict.values())

for (aa,n) in aa_dict.items():
    freq = n/total
    aa_freq[aa] = freq

print(aa_freq)

aa_freq_df = pd.DataFrame.from_dict(aa_freq, orient="index")
aa_freq_df = aa_freq_df.transpose()
aa_freq_df.to_csv("../../dataset/model/seq_peptide_data/bf_all.csv")












'''



#df1 = pd.read_csv('../../dataset/model/partition/f000_9mer.csv')
#df2 = pd.read_csv('../../dataset/model/partition/c000_9mer.csv')

#pos_df = pd.read_csv('../../dataset/model/seq_peptide_data/contigs/contigs_pos_9mer.txt', header=None)
#neg_df = pd.read_csv('../../dataset/model/seq_peptide_data/contigs/contigs_neg_9mer.txt', header=None)

#df = pd.concat([df1, df2])

#pos_df = df[df['target'] == 1]
#neg_df = df[df['target'] == 0]

#aa_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
aa_dict = {'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0,
           'G':0, 'H':0, 'I':0, 'J':0, 'K':0, 'L':0,  
           'M':0, 'N':0, 'P':0, 'Q':0, 'R':0, 'S':0, 
           'T':0, 'U':0, 'V':0, 'W':0, 'X':0, 'Y':0, 'Z':0}


print("Positive list: ")

with open("../../dataset/model/seq_peptide_data/contigs/contigs_pos_9mer.txt", 'r') as f:
    for line in f:
        line = line.strip()
        for n in range(len(line)):
            aa_dict[line[n]] += 1
            
print(aa_dict)


pos_df = pd.DataFrame.from_dict(aa_dict, orient="index")
pos_df = pos_df.transpose()
pos_df.to_csv("../../dataset/model/seq_peptide_data/contigs/bf_pos.csv")


aa_dict = {'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0,
           'G':0, 'H':0, 'I':0, 'J':0, 'K':0, 'L':0,
           'M':0, 'N':0, 'P':0, 'Q':0, 'R':0, 'S':0,
           'T':0, 'U':0, 'V':0, 'W':0, 'X':0, 'Y':0, 'Z':0}


print("Negative list: \n")

with open("../../dataset/model/seq_peptide_data/contigs/contigs_neg_9mer.txt", 'r') as f:
    for line in f:
        line = line.strip()
        for n in range(len(line)):
            aa_dict[line[n]] += 1

print(aa_dict)


neg_df = pd.DataFrame.from_dict(aa_dict, orient="index")
neg_df = neg_df.transpose()
neg_df.to_csv("../../dataset/model/seq_peptide_data/contigs/bf_neg.csv")


#for aa in aa_list:
#    print(f"{aa} occurance = {pos_df['extended'].str.count(aa).sum()}")
#    print(f"{aa} occurance = {pos_df.iloc[:,0].str.count(aa).sum()}")

#print("\nNegtaive list: ")

#for aa in aa_list:
#    print(f"{aa} occurance = {neg_df['extended'].str.count(aa).sum()}")
#    print(f"{aa} occurance = {neg_df.iloc[:,0].str.count(aa).sum()}")

'''
