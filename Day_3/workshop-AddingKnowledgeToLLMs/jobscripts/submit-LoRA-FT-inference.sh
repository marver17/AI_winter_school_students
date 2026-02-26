#!/bin/bash
#SBATCH --nodes=1                    		  # 1 node
#SBATCH --gres=gpu:4				          # GPUs per node
#SBATCH --time=00:30:00               		  # time limit: 2h
#SBATCH --error=logs/logs-LoRA-FT-inference-%j.err          # standard error file
#SBATCH --output=logs/logs-LoRA-FT-inference-%j.out         # standard output file
#SBATCH --account=tra26_minwinsc 		  # project account
#SBATCH --partition=boost_usr_prod 		  # partition name
#SBATCH --qos=boost_qos_dbg         # Queue name (boost_qos_dbg)
#SBATCH --cpus-per-task=32			  # CPUS per task

# Conda init
__conda_setup="$('/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate conda environment
module purge
conda activate /leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/envs/adding-knowledge-to-llms-env

# Check python path points out inside the Conda environment
which python

# Define Paths for Data and Model
export DATA_PATH="/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/datasets/ChatDoctor-dataset/data/"
export MODEL_PATH="/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/models/gemma-2-2b-it"
export SAVE_PATH_LORA_MODEL="/leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/models/FT-models/test/LoRA_model_chatdoctor_gemma-2-2b-it"

# Define question to ask to the Base Model and QLoRA model.
export USER_QUERY="Hello, My husband is taking Oxycodone due to a broken leg/surgery. He has been taking this pain medication for one month. We are trying to conceive our second baby. Will this medication afect the fetus? Or the health of the baby? Or can it bring birth defects? Thank you."

# Run python Fine-Tuning QLoRA script
python pythonInferenceScripts/inference-LoRA-FT.py

# Training Done
sleep 2
echo "LoRA Inference OK"



