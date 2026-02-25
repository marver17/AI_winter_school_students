#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH --account=tra26_minwinsc
#SBATCH --nodes=xxxxxxxxxxxxxxxxxx
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --qos=qos_prio

### Set environment ###
module load cuda
source /leonardo_work/tra26_minwinsc/pyenvs/acceleratenv/bin/activate

export GPUS_PER_NODE=xxxxxxxxxxxxxxxxxxxxxxxxxx

export BNB_CUDA_VERSION=121
export WANDB_MODE=offline

echo GPUS_PER_NODE=$GPUS_PER_NODE
echo NNODES=$SLURM_NNODES

#### Set network #####
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export CONFIG_PATH= #$CONFIG_PATH

#### Define Launcher, Script and Training Args ####

export LAUNCHER="accelerate launch \
    --config_file $CONFIG_PATH \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port 6000 \
    --machine_rank $SLURM_PROCID \
    "
export SCRIPT="FFT_DDP.py"
export CMD="$LAUNCHER $SCRIPT" # $TRAIN_ARGS"
echo "$CMD"

#### Launch it!! ####
srun $CMD

