#!/bin/bash
#SBATCH --nodes=1                    		  # 1 node
#SBATCH --gres=gpu:4				          # GPUs per node
#SBATCH --time=00:30:00               		  # time limit: 2h
#SBATCH --error=logs/logs-RAG-%j.err          # standard error file
#SBATCH --output=logs/logs-RAG-%j.out         # standard output file
#SBATCH --account=tra26_minwinsc 		  # project account
#SBATCH --partition=boost_usr_prod 		  # partition name
#SBATCH --qos=boost_qos_dbg         # Queue name (boost_qos_dbg)
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
export SAVE_PATH_FAISS_INDEX="$CINECA_SCRATCH/workshop-AddingKnowledgeToLLMs/FT-models/RAG/RAG-faiss-index.idx"
export SENTENCE_TRANSFORMER_MODEL_PATH="/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/models/multi-qa-MiniLM-L6-cos-v1"

# Run python Fine-Tuning QLoRA script
python pythonFTScripts/Faiss-RAG.py

# FAISS Index defined
sleep 2
echo "Faiss RAG Ended"


