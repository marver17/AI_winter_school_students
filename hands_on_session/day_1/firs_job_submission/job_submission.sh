#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpu-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=49400         ##mems per node MB
#SBATCH --account=tra26_minwinsc ###account
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_woas_dbg                   ####quality of service



module load python/3.11.7
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2 
srun python hello_world.py
