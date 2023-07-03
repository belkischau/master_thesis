import pandas as pd

# Read the CSV file into a pandas dataframe
included_peptide = pd.read_csv('../../dataset/included_peptide_extended.csv')
not_included_peptide = pd.read_csv('../../dataset/not_included_peptide_extended.csv')
iedb_not_included = pd.read_csv('../../dataset/iedb_not_included_peptide_extended.csv')
# raw = pd.read_csv('../../dataset/01a_success_epitope_protein_2.csv') 


# Create a new dataframe with only the unique pairs of protein and peptide
#unique_pairs = df[['protein', 'peptide']].drop_duplicates()
unique_pairs_included = included_peptide[['protein', 'peptide', 'before', 'after', 'extended']].drop_duplicates()
unique_pairs_not_included = not_included_peptide[['protein', 'peptide', 'before', 'after', 'extended']].drop_duplicates()
unique_pairs_iedb_not_included = iedb_not_included[['protein', 'peptide', 'before', 'after', 'extended']].drop_duplicates()


# Write the unique pairs to a new CSV file
unique_pairs_included.to_csv('../../dataset/included_peptide_extended_unique_pairs.csv', index=False)
unique_pairs_not_included.to_csv('../../dataset/not_included_peptide_extended_unique_pairs.csv', index=False)
unique_pairs_iedb_not_included.to_csv('../../dataset/iedb_not_included_peptide_extended_unique_pairs.csv', index=False)

