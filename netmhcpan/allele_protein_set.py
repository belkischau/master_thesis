
#!/usr/bin/env python

import os
#import subprocess
import pandas as pd


df = pd.read_csv("allele_protein.csv")


proteins = list()
alleles = list()
count_list = list()

for index, row in df.iterrows():
    allele = row['allele']
    alleles.append(allele)
    p = set(row['proteins'].split(','))
    count_list.append(len(p))
    p = ','.join(p)
    proteins.append(p)


#data = pd.DataFrame({
#    'allele': alleles, 
#    'proteins': proteins
#})

count_df = pd.DataFrame({
    'allele': alleles, 
    'count': count_list
})

#data.to_csv("netmhcpan/allele_protein_set.csv", index = None, sep = ',')
count_df.to_csv("allele_protein_set_count.csv", index = None, sep = ',')
