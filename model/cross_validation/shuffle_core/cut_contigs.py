import os 

path = "peptide_out_32"

files = os.listdir(path)

for file in files:
    file_name = file[:-4]
    cmd = f"cut -c 1-6,16-21 {path}/{file} > ../../../dataset/model/seq_peptide_data/shuffle_contigs/contigs_{file_name}.txt"
#    print(cmd)
    os.system(cmd)

