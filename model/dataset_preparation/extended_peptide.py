import csv
import os
import pandas as pd


in_file = "../../dataset/included_peptide_3_unique_pair.csv"
out_file = "../../dataset/included_peptide_3_extended.csv"
protein_directory = "../../data/fastas_with_header/"

protein_list = list()
peptide_list = list() 
before_list = list()
after_list = list()
extended_list = list()


#n = 0 
with open(in_file, 'r') as input_file:
    first_line = input_file.readline()

    for line in input_file:
        line = line.strip()
        # Get the peptide sequence
        peptide = line.split(',')[1]
#        print("Peptide: ", peptide)

        # Get the protein sequence
        protein = line.split(',')[0]
#        print("Protein: ", protein) 

        try: 
            # Load the protein sequence from the corresponding file
            with open(os.path.join(protein_directory, protein + '.fasta'), 'r') as protein_file:
                #protein_sequence = protein_file.read()
                protein_sequence = ''
                header = protein_file.readline()
                for l in protein_file: 
                    protein_sequence += l.strip()
#            print(protein_sequence)

        except Exception as e: 
            print("Eror occur: ", e)
            pass

        # Find the index of the peptide sequence in the protein sequence
        peptide_index = protein_sequence.find(peptide)
#        print(peptide_index) 

        if peptide_index == -1:
            print(peptide, " not found. ") 

        # Extract the 6 strings before and after the peptide sequence
        before = protein_sequence[peptide_index-6:peptide_index]
        after = protein_sequence[peptide_index+len(peptide):peptide_index+len(peptide)+6]
        extended = str(before + peptide + after)

#        print("Before: ", before) 
#        print("After: ", after) 
#        print("Extended: ", extended)

        protein_list.append(protein)
        peptide_list.append(peptide)
        before_list.append(before)
        after_list.append(after)
        extended_list.append(extended)

#        n += 1
#        if n == 5: 
#            break




data = pd.DataFrame({
    'protein': protein_list, 
    'peptide': peptide_list, 
    'before': before_list, 
    'after': after_list, 
    'extended': extended_list
})

#print(data)

#data.to_csv("../dataset/included_peptide_extended.csv", sep = ',', index=None)
        
data.to_csv(f'{out_file}', sep = ',', index=None)

