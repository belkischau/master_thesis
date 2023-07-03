#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from joblib import Parallel, delayed
import os.path
from os import path


# creating list of jobs
def makejoblist(allele, proteins, basic_cmd):
    joblist = list()
    for i in range(len(proteins)):
        p = proteins[i] 
#        print(p)
        file = "/home/projects/vaccine/people/belcha/master_thesis/data/fastas_with_header/" + str(p) + '.fasta'
#        return p
        outfile_name = "/home/projects/vaccine/people/belcha/master_thesis/netmhcpan/out/" + allele + '_' + str(p) + '.out'

        if path.exists(outfile_name): 
            pass 
        else: 
            netMHCpan_cmd = basic_cmd.format(filename = file, HLA_allele = allele, outfile = outfile_name)
            joblist.append(netMHCpan_cmd)
    return joblist


# run the job through subprocess
#def runjob(command):
#    # Don't need no output
#    job = subprocess.run(command.split())
# run the job through subprocess

def runjob(command):
    # Don't need no output
    outfile_name = command.split()[-1]
    
    with open(outfile_name, 'w') as output_file:
        job = subprocess.run(command.split(), stdout = output_file, text = True)

success_df = pd.read_csv("../dataset/01a_success_epitope_protein_2.csv")

statuses = success_df['status']
proteins= success_df['protein']
MHC_alleles = success_df['MHC_allele']

rows = list()

for i in range(len(statuses)): 
    if statuses[i] == "Successful": 
        rows.append(i)


MHC_allele_dict = dict()

for n in rows: 
    MHC_allele = MHC_alleles[n]
    protein = proteins[n]
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





#MHC_dict_df = pd.read_csv("/home/projects/vaccine/people/belcha/master_thesis/netmhcpan/MHC_dict.csv")

# MHC_dict = MHC_dict_df.to_dict('list')

#MHC_dict = dict([(a, [b]) for a, b in zip(MHC_dict_df['allele'], MHC_dict_df['proteins'])])

#n = 0 
#for i in MHC_dict.items(): 
#    print(i[0], i[1])
#    n += 1
#    if n == 5: 
#        break

netMHCpan_allele_list_file = "/home/projects/vaccine/people/belcha/master_thesis/netmhcpan/netMHCpan_MHC_list.txt"

netMHCpan_allele_list = list()

try: 
    with open(netMHCpan_allele_list_file) as f: 
        flag = False
        for line in f: 
            if flag: 
                netMHCpan_allele_list.append(line.split('\n')[0])
            if line.startswith('# Tmpdir made'): 
                flag = True
except IOError as error: 
    print("Cannot open file, reason: ", error)

#print(netMHCpan_allele_list[:5])

basic_cmd = "/home/projects/vaccine/people/s132421/CD19_CD22_project/netMHCpan-4.0/Linux_x86_64/bin/netMHCpan -f {filename} -a {HLA_allele} -BA -s > {outfile}"

#print(basic_cmd) 

n = 0 

not_found_allele = list()

for i in MHC_allele_dict.items(): 
    allele = i[0]
#    print(allele)
#    print(proteins)
    if '*' in allele: 
        allele = allele.replace('*', '')
    print("allele: ", allele)
    if allele in netMHCpan_allele_list: 
#        print(allele, " found in the list. ")
        proteins = i[1]
        joblist =  makejoblist(allele, proteins, basic_cmd)
        Parallel(n_jobs=20)(delayed(runjob)(x) for x in joblist)
#    
    else: 
        not_found_allele.append(allele)
        print(allele, " cannot be found in the list. ")

    n += 1
    print("Progress: ", n, " out of ", len(MHC_allele_dict), ".") 

not_found_file = "/home/projects/vaccine/people/belcha/master_thesis/netmhcpan/not_found_allele.out" 

with open(not_found_file, "w") as f:
    f.write(not_found_allele)

