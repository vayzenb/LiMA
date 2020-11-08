#!/bin/bash -l

# Job name
#SBATCH --job-name=LiMA_MSL  
# Mail events (NONE, BEGIN, END, FAIL, ALL)
###############################################
########## example #SBATCH --mail-type=END,FAIL 
##############################################
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
 
# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0

# Job memory request
#SBATCH --mem=20gb

# Time limit days-hrs:min:sec
#SBATCH --time 8:00:00

# Standard output and error log
#SBATCH --output=LiMA_MSL.out

python LiMA_MSL_SVM.py
