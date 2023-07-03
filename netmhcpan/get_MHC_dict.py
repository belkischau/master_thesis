#!/usr/bin/env python 

import pandas as pd
import os

# mhc_ligand_df = pd.read_csv("../raw_data/mhc_ligand_table_export_1676030517.csv", skiprows = 1, nrows = 10)

success_df = pd.read_csv("../dataset/01a_success_epitope_protein_2.csv")

outfile = "/home/projects/vaccine/people/belcha/master_thesis/netmhcpan/MHC_dict.csv"

# MHC_allele = mhc_ligand_df['Allele Name']

# success_df.insert(loc = 5, column = "MHC_allele", value = MHC_allele)

# print("Successfully add allele column. ")

# urls = [(row['Parent Protein IRI'], row['Description']) for index, row in mhc_ligand_df.iterrows()]

statuses = success_df['status']
proteins= success_df['protein']
MHC_alleles = success_df['MHC_allele']

#print("status: ", statuses[:5]) 
#print("proteins: ", proteins[:5]) 
#print("MHC: ", MHC_allele[:5])

rows = list()

for i in range(len(statuses)): 
    if statuses[i] == "Successful": 
        rows.append(i)

#print("rows: ", rows)

MHC_allele_dict = dict()

for n in rows: 
    MHC_allele = MHC_alleles[n]
#    print("allele: ", MHC_allele)
    protein = proteins[n]
#    print(MHC_allele_dict.keys())
#    print("protein: ", protein)
    if MHC_allele in MHC_allele_dict.keys(): 
        MHC_allele_dict[MHC_allele].append(protein)
    else:
        MHC_allele_dict[MHC_allele] = list()
        MHC_allele_dict[MHC_allele].append(protein)


dict_keys = list()
dict_values = list()

for i in MHC_allele_dict.items(): 
    dict_keys.append(i[0])
    dict_values.append(i[1])

df = pd.DataFrame({
    'allele': dict_keys, 
    'proteins': dict_values
})

df.to_csv(outfile, index = None, sep = ',')

#    with open(outfile, "w") as f:
#        f.write(i[0])
#        f.write(i[1])



