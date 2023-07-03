#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=vaccine -A vaccine
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N netmhcpan
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e netmhcpan.err
#PBS -o netmhcpan.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=20 
### Memory
#PBS -l mem=100gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 12 hours)
#PBS -l walltime=99:00:00


# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $pwd
cd /home/projects/vaccine/people/belcha/master_thesis/netmhcpan

### Here follows the user commands:
# Define number of processors
#NPROCS=`wc -l < $PBS_NODEFILE`
#echo This job has allocated $NPROCS nodes

# Load all required modules for the job
module load tools
module load anaconda3/2021.05
#module load anaconda3/2021.11
#module load anaconda3/4.4.0
#module load anaconda3/4.0.0


pip3 install --user pandas

export NETMHCpan=/home/projects/vaccine/people/s132421/CD19_CD22_project/netMHCpan-4.0/Linux_x86_64/


# This is where the work is done
# Make sure that this script is not bigger than 64kb ~ 150 lines, 
python3 get_possible_peptide2.py


echo "Finished blast! "
