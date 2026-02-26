#!/bin/bash
#SBATCH --nodes=1                    		  # 1 node
#SBATCH --gres=gpu:4				          # GPUs per node
#SBATCH --time=00:30:00               		  # time limit: 30 minutes
#SBATCH --error=logs/logs-Full-FT-%j.err          # standard error file
#SBATCH --output=logs/logs-Full-FT-%j.out         # standard output file
#SBATCH --account=tra26_minwinsc 		  # project account
#SBATCH --partition=boost_usr_prod 		  # partition name
#SBATCH --qos=boost_qos_dbg			  # Queue name
#SBATCH --cpus-per-task=32			  # CPUS per task

module purge
module load profile/deeplrn
module load cuda/12.1 nccl/2.19.1-1--gcc--12.2.0-cuda-12.1 cineca-ai

nvidia-smi

echo "OK"

sleep 5


