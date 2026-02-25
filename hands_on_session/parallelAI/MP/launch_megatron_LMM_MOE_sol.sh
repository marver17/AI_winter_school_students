#!/bin/bash
#SBATCH --account=tra26_minwinsc
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=training-megatron
##SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:05:00
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4


module load cuda


export CUDA_DEVICE_MAX_CONNECTIONS=1 
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# === Singularity Image Path ===
export PATH_SINGULARITY="/leonardo_work/tra26_minwinsc/containers/nemo_25.11.01.sif"

# === Host Folder Bind Mount Setup ===
export PATH_TOKENIZER="/leonardo_scratch/large/userinternal/dbrandon/T41_oeLLM/distributed-training-benchmarking-main/megatron-lm/leonardo/google/gemma-3-27b-it/"
export PATH_DATA="/leonardo_work/tra26_minwinsc/DATA"
export PATH_TO_BIN="FW/fineweb-10BT_text_document" # path inside the data folder. FineWeb 10BT subset available here: https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-10BT
export PATH_RESULTS="$CINECA_SCRATCH/results_megatron"
export PATH_LOGS="$CINECA_SCRATCH/logs_megatron"
export PATH_CACHE="$CINECA_SCRATCH/cache_megatron"

mkdir -p "$PATH_RESULTS" "$PATH_LOGS" "$PATH_CACHE"

# === Megatron Parallelism Params ===
TP=4
PP=4
CP=1
EP=4
# === Megatron Parallelism Params ===

# === Config Params ===
MBS=2
GBS=256
SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
TOTAL_ITERS=10
TRAIN_SAMPLES=$((TOTAL_ITERS*GBS))
LR_WARMUP_ITERS=50 #only for longer trainings 
LR_WARMUP_SAMPLES=$((LR_WARMUP_ITERS*GLOBAL_BATCH_SIZE)) 
LR_DECAY_SAMPLES=$TRAIN_SAMPLES

TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-tokenizer.model}" 
TE_FP8=0
FSDP=0

LOG_INTERVAL=1
SAVE_INTERVAL=500 
EVAL_INTERVAL=5000 
EVAL_ITERS=10

CKPT_FORMAT="${CKPT_FORMAT:-torch}"




# === Model Hyperparameters, to set them properly have a look at  ===
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_LAYERS=32
NUM_HEADS=32
INIT_METHOD_STD=0.008
MOE=1 #1 for MoE , 0 for dense models
NUM_EXPERTS=8
MOE_ROUTER_TOPK=2
MOE_ROUTER_LOAD_BALANCING_TYPE=aux_loss
MOE_AUX_LOSS_COEFF=0.01



# === Set TP and FSDP ===
if [[ "$FSDP" -eq 1 && "$TP" -gt 1 ]]; then
    echo "FSDP and TP are not compatible. Setting TP=1."
    export TP=1
fi

# Distributed args
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export NNODES=${#nodes_array[@]}
export GPUS_PER_NODE=4
export WORLD_SIZE=$NNODES*$GPUS_PER_NODE
echo "###NNODES:$NNODES\n"

 
# === Megatron args ===

export DISTRIBUTED_ARGS="--rdzv_id=$RANDOM \
 --rdzv_backend=c10d \
 --rdzv_endpoint=$head_node_ip:29505 \
 --nnodes=$NNODES \
 --nproc_per_node=$GPUS_PER_NODE"

export GPT_ARGS="\
    --use-mcore-models \
	--tensor-model-parallel-size ${TP} \
	--sequence-parallel \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
	--expert-model-parallel-size ${EP} \
	--expert-tensor-parallel-size 1 \
    --use-distributed-optimizer \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --make-vocab-size-divisible-by 128 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --swiglu \
    --disable-bias-linear \
    --init-method-std "${INIT_METHOD_STD}" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-samples ${TRAIN_SAMPLES} \
    --bf16 \
	--use-flash-attn \
    --group-query-attention \
    --num-query-groups 8"



export TRAIN_ARGS=" \
  --lr-decay-samples $LR_DECAY_SAMPLES \
  --lr-warmup-samples $LR_WARMUP_SAMPLES \
  --lr 1.2e-5 \
  --min-lr 1.2e-6 \
  --lr-decay-style cosine \
  --clip-grad 1.0 \
  --weight-decay 0.1
"

export DATA_ARGS=" \
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model /tokenizer/ \
    --data-path /data/${PATH_TO_BIN} \
    --dataloader-type cyclic \
    --num-workers 8 \
    --data-cache-path /cache"

export OUTPUT_ARGS=" \
    --log-interval ${LOG_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --ckpt-format ${CKPT_FORMAT} \
    --log-throughput \
    --log-validation-ppl-to-tensorboard \
    --split 99,1,0 \
"
export EXTRA_ARGS=" \
	--distributed-backend nccl \
	--distributed-timeout-minutes 60 \
        --overlap-grad-reduce \
	--optimizer adam \
	--adam-beta1 0.9 \
	--adam-beta2 0.95"

if [[ "$MOE" -eq 1 ]]; then
	GPT_ARGS+=" \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --moe-router-load-balancing-type ${MOE_ROUTER_LOAD_BALANCING_TYPE} \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-layer-recompute \
    --no-mmap-bin-files \
    --overlap-param-gather \
    --overlap-grad-reduce"
fi


# === Logging ===
export WANDB_MODE=offline
export WANDB_DIR=/logs/wandb

#export LOGGING_ARGS="\
#    --tensorboard-dir /logs \
#    --wandb-project= \
#    --wandb-exp-name= \
#    --wandb-save-dir "

export CKPT_LOAD_ARGS=""  

# === Launch ===


srun -l singularity exec --nv -B "$PATH_TOKENIZER:/tokenizer,$PATH_DATA:/data,$PATH_RESULTS:/results,$PATH_LOGS:/logs,$PATH_CACHE:/cache, /leonardo_work/tra26_minwinsc/parallelAI/Megatron-LM:/workspace/megatron-lm" $PATH_SINGULARITY \
     torchrun $DISTRIBUTED_ARGS \
     /opt/megatron-lm/pretrain_gpt.py \
     $GPT_ARGS $DATA_ARGS $OUTPUT_ARGS $EXTRA_ARGS $TRAIN_ARGS $CKPT_LOAD_ARGS
#$LOGGING_ARGS $CKPT_LOAD_ARGS

