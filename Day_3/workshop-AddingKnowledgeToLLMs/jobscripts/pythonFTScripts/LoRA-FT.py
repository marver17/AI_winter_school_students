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
from trl import SFTTrainer, SFTConfig
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
# 2. Tokenizer
# =====================================================
# Define the model we want to fine tune.
model_path = os.getenv("MODEL_PATH", None)
model_name = str(model_path.split("/")[-1])

# Get Model TokenizerSAVE_PATH_LORA_MODEL
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model used for LoRA Fine-Tuning: {model_name}")


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
# 4. LoRA Fine-Tuning
# =====================================================
# Read LoRA Model
lora_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

# Find Linear modules
modules = ["q_proj", "v_proj"]
print("Modules:", modules)

# Define LoRA Config
lora_config = LoraConfig(
    r=8,                        # Try distinct Rank values
    lora_alpha=32,              # Try distinct lora_alpha values
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Get only the LoRA params to modify during training
lora_model = get_peft_model(lora_model, lora_config)
lora_model.print_trainable_parameters()

# Define Training Arguments
lora_args = TrainingArguments(
    # Throughput critical
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,

    # Training length
    num_train_epochs=3,

    # Optimizer
    learning_rate=5e-5, #2e-4,
    fp16=False,
    bf16=True,

    # Logging
    logging_strategy="epoch",
    warmup_steps=30,

    output_dir=os.getenv("SAVE_PATH_LORA_MODEL", None),
    save_total_limit=2,
    save_strategy="epoch",

    # Evaluation
    eval_strategy="no",
    #eval_steps=50,
    
    # System
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
)

# Trainer class
lora_trainer = SFTTrainer(
    model=lora_model,
    args=lora_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)

# Before training
torch.cuda.reset_peak_memory_stats()
print("Allocated before training:", torch.cuda.memory_allocated()/1e9, "GB")
print("Reserved before training:", torch.cuda.memory_reserved()/1e9, "GB")

# Train LoRA Model with the Medical Q&A data.
# After training, get peak memory usage
train_output = lora_trainer.train()

print("Peak Allocated during training:", torch.cuda.max_memory_allocated()/1e9, "GB")
print("Peak Reserved during training:", torch.cuda.max_memory_reserved()/1e9, "GB")
# =====================================================
# 4.1. LoRA Save Fine-Tuning Model
# =====================================================
# Save LoRA models - ChatDoctor
save_path_lora_ft_model = os.getenv("SAVE_PATH_LORA_MODEL", None)
lora_model.save_pretrained(save_path_lora_ft_model, save_serialization=True)
tokenizer.save_pretrained(save_path_lora_ft_model)
print(f"LoRA FT Model saved to: {save_path_lora_ft_model}")


