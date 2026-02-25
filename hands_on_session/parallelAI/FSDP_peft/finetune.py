# SetUp
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training,LoraConfig, get_peft_model
import time

# Accelerate initialization
accelerator = Accelerator()
device = accelerator.device #✨

# Argument Parser
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments via CLI")

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps",type=int,default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    return parser.parse_args()

args = parse_args()


if accelerator.is_main_process: print('Setup done ✅')

# Dataset  Loading
dataset = load_from_disk(args.dataset_path) 

if accelerator.is_main_process: print('Loading dataset: done ✅')

# Tokenizer Loading
model_id = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

if accelerator.is_main_process: print('Loading tokenizer: done ✅')

# Data pre-processing for Llama3 chat
def format_example_for_llama3(example):
    user_instruction = example['instruction']
    assistant_response = example['response']  

    messages = [
        {"role": "system", "content": "You are a helpful and courteous customer support assistant."},
        {"role": "user", "content": user_instruction},
        {"role": "assistant", "content": assistant_response}
    ]

    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) #single, long string, formatted according to the Llama 3.1 chat template, including all the special tokens like <|begin_of_text|>, <|start_header_id|>system<|end_header_id|>, etc.
    return example

processed_dataset = dataset.map(format_example_for_llama3, remove_columns=dataset.column_names)

if accelerator.is_main_process: print('Dataset preprocessing: done ✅')

# Explicit Tokenization
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding=False, max_length=4096)

processed_dataset = processed_dataset.map(tokenize, batched=True,num_proc=4)

if accelerator.is_main_process: print('Tokenization: done ✅')

# Split into train and test
split = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = (split["train"], split["test"])

# Model Quantization before loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage= torch.bfloat16,
    bnb_4bit_use_double_quant=True, # True Recommended for better stability
)


# Model loading
if accelerator.is_main_process: print('Loading model in progress ⏳')

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,  
)

if accelerator.is_main_process: print(f'Loading model: done ✅ ({model_id})')

#----- model GPUs occupancy ------------------
import GPUtil

def print_gpu_utilization():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU id: {gpu.id}, name: {gpu.name}")
        print(f"  Load: {gpu.load * 100:.1f}%")
        print(f"  Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

if accelerator.is_main_process: print_gpu_utilization()
#-----------------------------------------------

# LoRA
lora_config = LoraConfig(
    r=64, # LoRA attention dimension
    lora_alpha=16, # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Modules to apply LoRA to
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # None, all, or lora_only
    task_type="CAUSAL_LM", # For language modeling
)

model = get_peft_model(model, lora_config)
if accelerator.is_main_process: print('LoRA config: done ✅')
model.print_trainable_parameters()
model.config.use_cache=False

#model = accelerator.prepare(model)

# Training
training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_train_batch_size, # start with 1 or 4, then use nvidia-smi ad adapt to best memory usage. 32 OoM
    gradient_accumulation_steps=1,  # 1 is ok, we don't have memory issues
    gradient_checkpointing=False, # for error 
    gradient_checkpointing_kwargs={'use_reentrant': False}, #for error
    optim="adamw_torch",
    learning_rate=5e-5, # usually the learning rate is lower for fine-tuning (between 1e-5 a 5e-5), but higher with LoRA (e-4)
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False,
    warmup_ratio=0.03,
    logging_steps=25,
    save_strategy="no",
    eval_strategy="no", #"steps"
    #eval_steps=100,
    bf16=True,
    remove_unused_columns=False,
    prediction_loss_only = True,
)

#-------- useful logs ----------------------------------------------
if accelerator.is_main_process:
        print("Model:", model_id)
        print("Parallelization technique: FSDP")
        print("GPU Batch Size:", training_args.per_device_train_batch_size)
        print("Learning rate:", training_args.learning_rate)
        print("Max steps:", training_args.max_steps)
        print("Epochs:", training_args.num_train_epochs)
        print("Gradient Accumulation steps:", training_args.gradient_accumulation_steps)
#-----------------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

s=time.time()
trainer.train() 
e=time.time()

if accelerator.is_main_process: print('Training: done ✅')
if accelerator.is_main_process: print(f'Finetuning completed in {e-s} seconds')

#output_path="./FFT_Llama_3.2_1B_Instructed"
#trainer.save_model(output_path)
#print(f"Final fine-tuned model saved to {output_path}")

