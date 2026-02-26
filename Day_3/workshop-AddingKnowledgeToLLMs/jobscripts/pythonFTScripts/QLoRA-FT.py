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
from peft import prepare_model_for_kbit_training


# Define Environment Variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

gpu_mem = get_gpu_memory()
print(gpu_mem)


# =====================================================
# 1. Load ChatDoctor Dataset
# =====================================================
# Load the dataset from the local directory
chatdoctor = load_dataset(os.environ.get("DATA_PATH"))

# =====================================================
# 2. Tokenizer
# =====================================================
# Define the model we want to fine tune.
model_path = os.environ.get("MODEL_PATH")
model_name = str(model_path.split("/")[-1])

# Get Model Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model used for QLoRA Fine-Tuning: {model_name}")


# =====================================================
# 3. Apply Chat Template & Data Collator with Dynamic Padding
# =====================================================
def format_chat_template(row):
    row_json = [{"role": "user", "content": f"INSTRUCTION:\n{row['instruction']}\n\nPATIENT MESSAGE:\n{row['input']}"},
                {"role": "assistant", "content": row["output"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Apply chat template to all data
chatdoctor = chatdoctor.map(format_chat_template, num_proc=1)

# Get train dataset
train_dataset = chatdoctor['train']

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

    # Move everything to GPU 0.
    #tokenized = {k: v.to('cuda:0') for k, v in tokenized.items()}
    
    return tokenized

# Subsample for workshop
train_data = train_dataset.select(range(3000)) #.shuffle(seed=42).select(range(2000))
#val_data = val_dataset.select(range(300))


# =====================================================
# 4. QLoRA (Quantized + LoRA)
# =====================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# Get Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Load weights in 4-bit precision instead of the usual 16-bit
    #bnb_4bit_compute_dtype=torch.bfloat16, # data type used for computations after quantization.
    #bnb_4bit_use_double_quant=True,        # double quantization, a technique to reduce quantization error (re-quantized with a second small scale factor)
    bnb_4bit_quant_type="nf4"              # nf4 stands for NormalFloat 4-bit (nf4 uses nonlinear mapping)
)

# Base model quantized to 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    #load_in_4bit=True,
    quantization_config=bnb_config,
    device_map={'': 0},
    torch_dtype=torch.bfloat16  # optional for LoRA
)

modules = ["q_proj", "v_proj"]
print("Modules:", modules)

peft_config = LoraConfig(
    r=2,
    lora_alpha=8,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Get LoRA Config in the Quantized Model
model = get_peft_model(model, peft_config)

modules = ["q_proj", "v_proj"]
print("Modules:", modules)

# Define Training Arguments
qlora_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    
    optim="paged_adamw_32bit",
    
    output_dir=os.environ["QLORA_FT_MODEL_PATH"],
    save_total_limit=1,
    save_strategy="epoch",
    
    logging_strategy="epoch",
    warmup_steps=30,
    
    learning_rate=5e-5,
    fp16=False,
    bf16=False,

    # System
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
)

# Trainer class
qlora_trainer = SFTTrainer(
    model=model,
    args=qlora_args,
    train_dataset=train_data,
    data_collator=data_collator,
)

# Before training
torch.cuda.reset_peak_memory_stats()
print("Allocated before training:", torch.cuda.memory_allocated()/1e9, "GB")
print("Reserved before training:", torch.cuda.memory_reserved()/1e9, "GB")

# Train qLoRA Model with the Medical Q&A data.
# After training, get peak memory usage
qlora_trainer.train()

print("Peak Allocated during training:", torch.cuda.max_memory_allocated()/1e9, "GB")
print("Peak Reserved during training:", torch.cuda.max_memory_reserved()/1e9, "GB")


# =====================================================
#    4.1. Save QLoRA Fine-Tuning Model
# =====================================================
# FT QLoRA Model - ChatDoctor
qlora_model_chatdoctor = qlora_trainer.model

# Save QLoRA Model - ChatDoctor
save_path_qlora_ft_model = os.getenv("SAVE_PATH_QLORA_MODEL", None)
qlora_model.save_pretrained(save_path_qlora_ft_model, save_serialization=True)
tokenizer.save_pretrained(save_path_qlora_ft_model)

print(f"QLoRA FT Model saved to: {save_path_qlora_ft_model}")

