# =====================================================
# 0. Setup
# =====================================================
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from utils.utils import get_gpu_memory, generate_chat_response
import bitsandbytes as bnb
import torch.nn as nn

# Define Environment Variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

gpu_mem = get_gpu_memory()
print(gpu_mem)

# =====================================================
# 1. Load ChatDoctor Dataset
# =====================================================
# Load the dataset from the local directory
chatdoctor = load_dataset(os.getenv("DATA_PATH", None))
# =====================================================
# 2. Model Path & Tokenizer
# =====================================================
# Define the model we want to fine tune.
model_path = os.getenv("MODEL_PATH", None)
model_name = str(model_path.split("/")[-1])

# Get Model Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model used for Full Fine-Tuning: {model_name}")

# =====================================================
# 3. Apply Chat Template & Data Collator with Dynamic Padding
# =====================================================
def format_chat_template(row):
    row_json = [
        {"role": "user", "content": f"INSTRUCTION:\n{row['instruction']}\n\nPATIENT MESSAGE:\n{row['input']}"},
        {"role": "assistant", "content": row["output"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Apply chat template to all data
chatdoctor = chatdoctor.map(format_chat_template, num_proc=4)

# Split Train and Test datasets
split_dataset = chatdoctor['train'].train_test_split(
    test_size=0.20,
    seed=42,
    shuffle=True,
)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

# Define the Data Collator for creating batches of the data
def data_collator(batch):
    tokenized = tokenizer(
        [example["text"] for example in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    # For causal LM, labels are just input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Subsample for workshop (select only X rows)
train_data = train_dataset.select(range(3000)) #.shuffle(seed=42).select(range(2000)) # Shuffle before choosing X rows
val_data = val_dataset.select(range(300))

# =====================================================
# 4. Read Model + Full Fine-Tuning
# =====================================================
# Read Base Model
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

# Define Training Arguments
train_args = TrainingArguments(
    per_device_train_batch_size=2,    # Try distinct values of batch_size
    gradient_accumulation_steps=8,    # Try distinct values of gradients accumulation
    num_train_epochs=3, 
    learning_rate=5e-5, #2e-4,        # Try distinct learning_rate values
    fp16=False,
    bf16=True,
    logging_strategy="steps",
    logging_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    warmup_steps=30,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)

# Trainer class
full_trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)

# Before training
torch.cuda.reset_peak_memory_stats()
print("Allocated before training:", torch.cuda.memory_allocated()/1e9, "GB")
print("Reserved before training:", torch.cuda.memory_reserved()/1e9, "GB")

# Train Full Model with the Medical Q&A data.
# After training, get peak memory usage
full_trainer.train()

print("Peak Allocated during training:", torch.cuda.max_memory_allocated()/1e9, "GB")
print("Peak Reserved during training:", torch.cuda.max_memory_reserved()/1e9, "GB")

# =====================================================
#    4.1. Save Full Fine-Tuning Model
# =====================================================
# FT Full Model - ChatDoctor
full_ft_model_chatdoctor = full_trainer.model

# Save Full Model - ChatDoctor
save_path_full_ft_model = os.getenv("SAVE_PATH_FULL_MODEL", None)
full_ft_model_chatdoctor.save_pretrained(save_path_full_ft_model)
tokenizer.save_pretrained(save_path_full_ft_model)

print(f"Full FT Model saved to: {save_path_full_ft_model}")

