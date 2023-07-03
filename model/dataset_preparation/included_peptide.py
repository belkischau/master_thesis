#!/usr/bin/env python

import os
#import subprocess
import pandas as pd


#df = pd.read_csv("dataset/01a_success_epitope_protein_2.csv")

protein_epitope_dict = dict()


#for index, row in df.iterrows():
#    if row['status'] == "Successful": 
#        epitope = row['epitope']
#        protein = row['protein']
#        if protein in protein_epitope_dict.keys():
#            protein_epitope_dict[protein].append(epitope)
#        else: 
#            protein_epitope_dict[protein] = list()
#            protein_epitope_dict[protein].append(epitope)


with open("../../dataset/01a_success_epitope_protein_2.csv", 'r') as infile: 
    for line in infile: 
        if line.split(',')[2] == "Successful": 
            epitope = line.split(',')[0]
            protein = line.split(',')[1]
            if protein == "H0YMF3":
                protein = 'H0YN83'
            if protein == "E9PJD7":
                protein = "P0DTL6"
            if protein in ["A0A1B0GUL6", "Q9UPP5", "A0A1B0GWG1", "O14767", "Q6V9S5", "Q8N3X7"]:
                protein == "Q5XKL5"



            if protein in protein_epitope_dict.keys():
                protein_epitope_dict[protein].append(epitope)
            if protein not in protein_epitope_dict.keys(): 
                protein_epitope_dict[protein] = list()
                protein_epitope_dict[protein].append(epitope)



#all_out_df = pd.read_csv("netmhcpan/out/new_SB_WB_list.out", sep = '\t')

yes_peptide_list = list()
yes_protein_list = list()
no_peptide_list = list()
no_protein_list = list()
yes_rank_list = list()
no_rank_list = list()

with open("../../netmhcpan/out/SB_WB_list_clear.out", 'r') as f: 
    for line in f:
        if (line.split()[-1] == 'WB') or (line.split()[-1] == 'SB'):

            peptide = line.split()[2]
            protein = line.split()[10]
            rank = line.split()[12]
#        print(protein)
            try: 
                protein = protein.split('_')[1]
            except IndexError as error: 
                print("Cannot find protein: ", protein)
                break
            try: 
                if peptide in protein_epitope_dict[protein]:
                    yes_peptide_list.append(peptide)
                    yes_protein_list.append(protein)
#                    yes_rank_list.append(rank)
                else: 
                    no_peptide_list.append(peptide)
                    no_protein_list.append(protein)
#                    no_rank_list.append(rank)
            except KeyError as error: 
#            print("Cannot find peptide: ", peptide, "\n& reason: ", error) 
#            print(error, '\n', line)
                no_peptide_list.append(peptide)
                no_protein_list.append(protein)
#                no_rank_list.append(rank)

print("Number of peptide included: ", len(yes_peptide_list))
print("Number of peptide not included: ", len(no_peptide_list))

yes_df = pd.DataFrame({
    "protein": yes_protein_list, 
    "peptide": yes_peptide_list
#    "rank": yes_rank_list
})


#yes_df = yes_df[['protein', 'peptide', 'rank']].drop_duplicates()
yes_df = yes_df[['protein', 'peptide']].drop_duplicates()


#yes_df.to_csv("../../dataset/included_peptide_3_with_rank.csv", index = None, sep = ',')
yes_df.to_csv("../../dataset/included_peptide_3_unique_pair.csv", index = None, sep = ',')

no_df = pd.DataFrame({
    "protein": no_protein_list, 
    "peptide": no_peptide_list 
#    "rank": no_rank_list
})

#no_df = no_df[['protein', 'peptide', 'rank']].drop_duplicates()
no_df = no_df[['protein', 'peptide']].drop_duplicates()

#no_df.to_csv("../../dataset/not_included_peptide_3_with_rank.csv", index = None, sep = ',')
no_df.to_csv("../../dataset/not_included_peptide_3_unique_pair.csv", index = None, sep = ',')
