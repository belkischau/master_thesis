import os
import pandas as pd 
import requests
import concurrent.futures


# ============ function ==================


def get_fasta(iri, dest, epitope):
    baseUrl = "https://rest.uniprot.org/uniprotkb/"
    if len(str(iri).split("/uniprot/")) < 2: 
        return epitope, "NA", "Unknown", 0, 0
    else: 
        protein = str(str(iri).split("/uniprot/")[1])
    currentUrl = baseUrl + protein + ".fasta"
    
    try: 
        response = requests.post(currentUrl)
    except Exception: 
        return epitope, protein, "Fail", 0, 0

    cData=''.join(response.text)
    pSeq = "".join(cData.split("\n")[1:])
    fasta = '\n'.join(cData.split('\n'))

    start_pos = pSeq.find(epitope) 
    protein_length = len(pSeq)
    if start_pos == -1: 
        return epitope, protein, "Unmatched", 0, 0
    else: 
        file_name = os.path.join(dest, protein + '.fasta')
        with open(file_name, 'w') as f:
            f.write(fasta)
        return epitope, protein, "Successful", start_pos, protein_length

    # if epitope in pSeq: 
    #     file_name = os.path.join(dest, protein + '.fasta')
    #     with open(file_name, 'w') as f:
    #         f.write(fasta)
    #     return protein, "Successful"
    # else: 
    #     return protein, "Fail"




# ============ main ===============


cwd = os.getcwd()
dest = os.path.join(cwd, 'data/fastas_with_header')
dataset_path = os.path.join(cwd, 'dataset/')


mhc_ligand_df = pd.read_csv("raw_data/mhc_ligand_table_export_1676030517.csv", skiprows = 1)

if not os.path.exists(dest):
    os.makedirs(dest)

if not os.path.exists(dataset_path): 
    os.makedirs(dataset_path)


urls = [(row['Parent Protein IRI'], row['Description']) for index, row in mhc_ligand_df.iterrows()]


# create a thread pool and submit each URL for download
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(get_fasta, url, dest, epitope) for url, epitope in urls]

    # collect the lengths and search positions of the downloaded text in separate lists
    epitopes = []
    proteins = []
    statuses = []
    starting_positions = []
    protein_lengths = []

    for future in concurrent.futures.as_completed(futures):
        epitope, protein, status, pos, protein_length = future.result()
        epitopes.append(epitope)
        proteins.append(protein)
        statuses.append(status)
        starting_positions.append(pos)
        protein_lengths.append(protein_length)

MHC_allele = mhc_ligand_df['Allele Name']

data = pd.DataFrame({
    'epitope': epitopes,
    'protein': proteins, 
    'status': statuses, 
    'starting_position': starting_positions, 
    'protein_length': protein_lengths, 
    'MHC_allele': MHC_allele
    })

data.to_csv(f'{dataset_path}/01a_success_epitope_protein_2.csv', index = None, sep = ',')







# success_epitope = list()
# success_protein = list()
# failed_fasta = list()
# unmatched_epitope = list()
# unmatched_protein = list()
# unknown_protein = list()
# starting_position = list()
# protein_length = list()

# # n = 0

# urls = [(row['Parent Protein IRI'], row['Description']) for index, row in mhc_ligand_df.iterrows()]


# # create a thread pool and submit each URL for download
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(download_url, url) for url in urls]

#     # collect the lengths of the downloaded text in a list
#     lengths = [future.result() for future in concurrent.futures.as_completed(futures)]

# # for i in tqdm(range(0, 10) , desc = 'Progress to finish downloading all fasta files: '):
# for i in tqdm(range(mhc_ligand_df.shape[0]), desc = 'Progress to finish downloading all fasta files: '):
#     iri = mhc_ligand_df['Parent Protein IRI'][i]
#     epitope = mhc_ligand_df['Description'][i]
    
#     if pd.notna(iri): 
#         protein = str(str(iri).split("/uniprot/")[1])
#         file = protein + ".fasta"
#         # for j in tqdm(range(0, 100), desc = 'Downloading fasta: '):
#         try: 
#             results = get_fasta(iri, dest, epitope)
#             if results[1] == "Successful": 
#                 success_epitope.append(epitope)
#                 success_protein.append(results[0])
#                 starting_position.append(results[2])
#                 protein_length.append(results[3])
#             elif results[1] == "Fail": 
#                 unmatched_epitope.append(epitope)
#                 unmatched_protein.append(results[0])
#         except:
#             failed_fasta.append(epitope)
#             pass

#     # if file in os.listdir(dest):
#     #     success_epitope.append(epitope)
#     #     success_protein.append(protein)

#     # else: 
#     #     try: 
#     #         results = get_fasta(iri, dest, epitope)
#     #         if results[1] == "Successful": 
#     #             success_epitope.append(epitope)
#     #             success_protein.append(results[0])
#     #         elif results[1] == "Fail": 
#     #             unmatched_epitope.append(epitope)
#     #             unmatched_protein.append(results[0])
#     #     except:
#     #         failed_fasta.append(epitope)
#     #         pass
            
#     else: 
#         unknown_protein.append(epitope)
        
#     # n += 1 
#     # if n == 10:
#     #     break 


# if len(success_epitope) == len(success_protein) & len(success_epitope) == len(starting_position) & len(success_epitope) == len(protein_length): 
#     success_data = pd.DataFrame({
#         'epitope': success_epitope,
#         'protein': success_protein, 
#         'starting_position': starting_position, 
#         'protein_length': protein_length
#     })
# else:
#     print("Unmatched length of epitopes and proteins. Some proteins cannot be downloaded. ")

# success_data.to_csv(f'{dataset_path}/01a_success_epitope_protein.csv', index = None, sep = ',')

# if len(unmatched_epitope) == len(unmatched_protein): 
#     unmatched_data = pd.DataFrame({
#         'epitope': unmatched_epitope,
#         'protein': unmatched_protein
#     })
# else:
#     print("Unmatched length of epitopes and proteins. Missing epitopes or proteins. ")

# unmatched_data.to_csv(f'{dataset_path}/01b_unmatched_epitope_protein.csv', index = None, sep = ',')


# print("Number of successfully downloaded fasta files: ", len(success_epitope))
# print("Number of fail downloaded fasta file: ", len(failed_fasta))
# print("Number of unknown proteins: ", len(unknown_protein))
# print("Number of unmatched proteins: ", len(unmatched_protein))

