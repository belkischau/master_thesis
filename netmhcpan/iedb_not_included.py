#!/usr/bin/env python

import os
import pandas as pd


#df = pd.read_csv("dataset/01a_success_epitope_protein_2.csv")

protein_epitope_dict = dict()


with open("../dataset/included_peptide.csv", 'r') as infile: 
    for line in infile: 
        epitope = line.split(',')[1]
        epitope = epitope.strip() 
        protein = line.split(',')[0]
        
        if protein in protein_epitope_dict.keys():
            protein_epitope_dict[protein].append(epitope)

        if protein not in protein_epitope_dict.keys(): 
            protein_epitope_dict[protein] = list()
            protein_epitope_dict[protein].append(epitope)

print("Dict: ", len(protein_epitope_dict.keys()))       

#n = 0
#for x,y in protein_epitope_dict.items():
#    print(x, y)
#    if n == 10:
#        break


yes_peptide_list = list()
yes_protein_list = list()
no_peptide_list = list()
no_protein_list = list()


with open("../dataset/01a_success_epitope_protein_2.csv", 'r') as f: 
    for line in f:
        if line.split(',')[2] == "Successful": 
            peptide = line.split(',')[0]
            protein = line.split(',')[1]

            if protein == "H0YMF3":
                protein == 'H0YN83'
            if protein == "E9PJD7":
                protein == "P0DTL6"
            if protein in ["A0A1B0GUL6", "Q9UPP5", "A0A1B0GWG1", "O14767", "Q6V9S5", "Q8N3X7"]:
                protein == "Q5XKL5"

            try: 
                if peptide in protein_epitope_dict[protein]:
                    yes_peptide_list.append(peptide)
                    yes_protein_list.append(protein)
                else: 
                    no_peptide_list.append(peptide)
                    no_protein_list.append(protein)
            except KeyError as error: 
    #            print("Cannot find peptide: ", peptide, "\n& reason: ", error) 
    #            print(error, '\n', line)
                no_peptide_list.append(peptide)
                no_protein_list.append(protein)



print("Number of peptide included: ", len(yes_peptide_list))
print("Number of peptide not included: ", len(no_peptide_list))

print("Number of protein included: ", len(yes_protein_list))
print("Number of protein not included: ", len(no_protein_list))

no_df = pd.DataFrame({
    "protein": no_protein_list, 
    "peptide": no_peptide_list
})

no_df.to_csv("../dataset/iedb_not_included_peptide.csv", index = None, sep = ',')
