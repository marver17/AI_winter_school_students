#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH --account=tra26_minwinsc
#SBATCH --nodes=1
#SBATCH -p boost_usr_prod
#SBATCH --time 01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --exclude=lrdn[1090-3456]
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --qos=qos_prio

### Set environment ###

source /leonardo_work/tra26_minwinsc/pyenvs/acceleratenv/bin/activate
export WANDB_MODE=offline

export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$((${NNODES}*${GPUS_PER_NODE}))

#### Set network #####
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

#### Set experiment ####

export MODEL_PATH=""
export DATASET_PATH=""
export CONFIG_PATH="config_FSDP.yaml"
export EPOCHS=1
export MAX_STEPS=-1 # consider only EPOCHS
export GPU_BS=8

#### Define Launcher, Script and Training Args ####

export LAUNCHER="accelerate launch \
    --config_file $CONFIG_PATH \
    --rdzv_backend c10d \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port 6000 \
    --machine_rank $SLURM_PROCID \
    "

export SCRIPT="finetune.py"

export TRAIN_ARGS="--model_path=$MODEL_PATH \
                   --dataset_path=$DATASET_PATH \
                   --num_train_epochs=$EPOCHS \
                   --max_steps=$MAX_STEPS \
                   --per_device_train_batch_size=$GPU_BS "

export CMD="$LAUNCHER $SCRIPT $TRAIN_ARGS"
echo "$CMD"

#### Launch it!! ####
srun $CMD









