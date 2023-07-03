#!/usr/bin/env python

import os
#import subprocess
import pandas as pd
#from joblib import Parallel, delayed
import os.path
from os import path

batch_path = '../data/fastas_with_header/batch/'
outfile_path = 'out/batch/'
bash_script_path = 'bash_scripts/template.sh'


if not os.path.exists(batch_path):
    os.mkdir(batch_path)

if not os.path.exists(outfile_path):
    os.mkdir(outfile_path)



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

not_found_allele = list()

MHC_dict = pd.read_csv("allele_protein_set.csv")

netMHCpan_basic_cmd = '/home/projects/vaccine/people/morni/netMHCpan-4.1/netMHCpan -f {filename} -a {HLA_allele} -s -BA > {netMHCpan_outfile}'


bash_scripts = '#!/bin/sh \n#PBS -W group_list=vaccine -A vaccine \n#PBS -N {job_name} \n#PBS -e {job_name}.err \n#PBS -o {job_name}.log \n#PBS -m n \n#PBS -l nodes=1:ppn=1 \n#PBS -l mem=16gb \n#PBS -l walltime=6:00:00 \n'

bash_scripts += 'cd /home/projects/vaccine/people/belcha/master_thesis/netmhcpan \nmodule load tools \nmodule load anaconda3/2021.05 \n'


for index, row in MHC_dict.iterrows():
    allele = row['allele']
    if '*' in allele:
        allele = allele.replace('*', '')
    if allele in netMHCpan_allele_list: 
        protein_fastas = ''

        batch_no = 0


#        for i, protein in enumerate(row['proteins'].split(',')):
        for i in range(0, len(row['proteins'].split(',')), 2000):
#            batch_protein = row['proteins'].split(',')[i:i+2000]
            for protein in row['proteins'].split(',')[i:i+2000]: 

                fasta = "../data/fastas_with_header/" + protein + '.fasta ' 
                protein_fastas += fasta
            
#            if i % 1000 == 0 or i == len(row['proteins'].split(',')): 
            batch_no += 1
            cat_outfile = batch_path + str(allele) + '_batch_' + str(batch_no) + '.fasta'
            cat_cmd = 'cat ' + protein_fastas + ' > ' + cat_outfile + '\n'
                
            outfile = outfile_path + str(allele) + '_batch_' + str(batch_no) + '.out'
            netMHCpan_cmd = netMHCpan_basic_cmd.format(filename = cat_outfile, HLA_allele = allele, netMHCpan_outfile = outfile) + '\n'
            # new_bash_cmd = 'cp template.sh {HLA_allele}_{batch}.sh \n'
            jobname = '{HLA_allele}_{batch}'.format(HLA_allele = allele, batch = batch_no)
            new_bash_file = 'bash_scripts/' + jobname + '.sh'
            with open(new_bash_file, 'w') as f: 
                f.write(bash_scripts.format(job_name = jobname))
                f.write(cat_cmd)
                f.write(netMHCpan_cmd)

            cmd = 'qsub {scripts}'.format(scripts = new_bash_file)
            try: 
                os.system(cmd)
            except Exception as error: 
                print("Cannot submit files, reason: ", error)
            protein_fastas = ''
 

    else:
        not_found_allele.append(allele)
        print(allele, " cannot be found in the list. ")

#    if index == 1: 
#        break
            
with open("out/not_found_allele.txt", 'w') as f:
    f.write(not_found_allele)

