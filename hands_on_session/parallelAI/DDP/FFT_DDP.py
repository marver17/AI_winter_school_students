import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from accelerate import Accelerator
from datasets import load_from_disk
import time

# Accelerate initialization
accelerator = Accelerator()
device = accelerator.device 


# --- Configurazione ---
dataset_path= ""
model_id = ""

# Caricamento del tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# Caricamento del modello in 4-bit per QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # O torch.float16 se la tua GPU non supporta bfloat16
    device_map=device # Copy the model on each GPU
)
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

# --- 2. Preparazione dei Dati ---
dataset = load_from_disk(dataset_path) 

def format_example_for_llama3(example):
    user_instruction = example['instruction']
    assistant_response = example['response']  

    messages = [
        {"role": "system", "content": "You are a helpful and courteous customer support assistant."},
        {"role": "user", "content": user_instruction},
        {"role": "assistant", "content": assistant_response}
    ]

    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return example

processed_dataset = dataset.map(format_example_for_llama3, remove_columns=dataset.column_names)

train_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)["train"]
eval_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)["test"]


# --- 3. Configurazione Argomenti di Training ---
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs= ##################### COMPLETE HERE ##########################################
    per_device_train_batch_size= ####################### COMPLETE HERE ###########################
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    optim="adamw_torch", # Ottimizzatore standard per full fine-tuning
    learning_rate=2e-5, # Tasso di apprendimento tipicamente più basso per full fine-tuning (es. 1e-5 a 5e-5)
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False,
    warmup_ratio=0.03,
    logging_steps=25,
    save_strategy="no",
    eval_strategy="no",
    bf16=True,
    push_to_hub=False,
    report_to="none",
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

# --- 4. Inizializzazione e Avvio del SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args
)
if accelerator.is_main_process:
   print("Starting Full Fine-tuning...")
trainer.train()

if accelerator.is_main_process:
   print("Full Fine-tuning completed...")


# Libera la memoria GPU
del model
del trainer
torch.cuda.empty_cache()

