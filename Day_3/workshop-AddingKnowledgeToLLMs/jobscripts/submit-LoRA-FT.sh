#!/bin/bash
#SBATCH --nodes=1                    		  # 1 node
#SBATCH --gres=gpu:4				          # GPUs per node
#SBATCH --time=00:30:00               		  # time limit: 2h
#SBATCH --error=logs/logs-LoRA-FT-%j.err          # standard error file
#SBATCH --output=logs/logs-LoRA-FT-%j.out         # standard output file
#SBATCH --account=tra26_minwinsc 		  # project account
#SBATCH --partition=boost_usr_prod 		  # partition name
#SBATCH --qos=boost_qos_dbg               # Queue name (boost_qos_dbg)
#SBATCH --cpus-per-task=32			  # CPUS per task

# Activate environment
module purge
module load python
source /leonardo_work/tra26_minwinsc/pyenvs/venv_llm/bin/activate

# Check python path points out inside the Conda environment
which python

# Define Paths for Data and Model
export DATA_PATH="/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/datasets/ChatDoctor-dataset/data/"
export MODEL_PATH="/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/models/gemma-2-2b-it"
export SAVE_PATH_LORA_MODEL="$CINECA_SCRATCH/workshop-AddingKnowledgeToLLMs/FT-models/LoRA_model_chatdoctor_gemma-2-2b-it"

# Run python Fine-Tuning QLoRA script
python pythonFTScripts/LoRA-FT.py

# Training Done
sleep 2
echo "LoRA FT Ended"


