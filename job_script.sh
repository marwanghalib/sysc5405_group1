#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=5:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=marwanghalib@cmail.carleton.ca
#SBATCH --mail-type=ALL

cd /project/def-jrgreen/mghalib/group_1/sysc5405_group1
module load
source ~/py38-cc/bin/activate


python3 lstm_RNN_CV.py
