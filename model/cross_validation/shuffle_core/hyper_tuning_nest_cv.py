import os
import time


outer_loop = "outer"
inner_loop = "outer/inner"

if not os.path.exists(outer_loop):
    os.mkdir(outer_loop)
if not os.path.exists(inner_loop):
    os.mkdir(inner_loop)


for i in range(5):
    if not os.path.exists(f"{outer_loop}{i}"):
        os.mkdir(f"{outer_loop}{i}")
#    if not os.path.exists(f"{inner_loop}/{path}"):
#       	os.mkdir(f"{inner_loop}/{path}")
    script_path = f"{outer_loop}{i}/bash_scripts"
    outfile_path = f"{outer_loop}{i}/output"
    if not os.path.exists(script_path):
        os.mkdir(script_path)

    if not os.path.exists(outfile_path):
        os.mkdir(outfile_path)

    base_bash_scripts = '#!/bin/sh \n#PBS -W group_list=vaccine -A vaccine \n#PBS -N {job_name} \n#PBS -e err_log/{job_name}.err \n#PBS -o err_log/{job_name}.log \n#PBS -m n \n#PBS -l nodes=1:ppn=1 \n#PBS -l mem=16gb \n#PBS -l walltime=00:40:00 \n'

    base_bash_scripts += 'cd /home/projects/vaccine/people/belcha/master_thesis/model/cross_validation/shuffle_core \nmodule load tools \nmodule load anaconda3/2021.05 \n'



    lr_list = [0.01, 0.001, 0.0001, 0.00001]
    #op_list = ['Adam', 'SGD']
    dropout_list = [0.1, 0.2, 0.3]
    hidden1_list = [16, 32, 64, 128]
    hidden2_list = [4, 8, 16, 32]
    batch_size_list = [32, 64]

    all_files = ['c000', 'c001', 'c002', 'c003', 'c004']

    test_file = all_files[i]

    inner_files = all_files.copy()
    inner_files.remove(all_files[i])
    print(f"test file = {test_file}")

    for n in range(len(inner_files)): 
    #    train_files = inner_files[n]
        valid_file = inner_files[n]
        train_files = inner_files.copy()
        train_files.remove(valid_file)
        train_files = ', '.join(train_files)
        print(f"valid file = {valid_file}, train file = {train_files}")
        print(f"'{train_files}'")



        for a in lr_list:
            lr = a
            for b in batch_size_list:
                batch_size = b
                for c in dropout_list:
                    dropout = c
                    for d in hidden1_list:
                        h1 = d
                        for e in hidden2_list:
                            h2 = e
                            job = f"{test_file}_{valid_file}_{a}_{b}_{c}_{d}_{e}"
                            command_line = f"time python3 model4_hp_inner_cv.py -train '{train_files}' \
                                -valid {valid_file} -test {test_file} -lr {a} -b {b} -d {c} -h1 {d} -h2 {e} > {outfile_path}/{job}.out"
                            script_file = f"{script_path}/{job}.sh"
                            with open(script_file, 'w') as f: 
                                f.write(base_bash_scripts.format(job_name = job))
                                f.write(command_line)
                            cmd = f'qsub {script_file}'
                            try: 
                                os.system(cmd)
                            except Exception as error: 
                                print("Cannot submit files, reason: ", error)
                            time.sleep(5)
#                            break
#                        break
#                    break
#                break
#            break
#        break
#    break


