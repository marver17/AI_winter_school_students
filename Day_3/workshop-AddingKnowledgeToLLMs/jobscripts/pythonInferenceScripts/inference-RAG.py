# =====================================================
# 7. Inference with Base Model and RAG
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
from utils.utils import get_gpu_memory, generate_chat_response, generate_RAG_response
import bitsandbytes as bnb
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import faiss
import random
from bert_score import score as bert_score_function
from rouge_score import rouge_scorer as rouge_scorer_function
import numpy as np
from transformers import logging

# Define the model we want to fine tune.
model_path = os.getenv("MODEL_PATH", None)
model_name = str(model_path.split("/")[-1])
device = "cuda"

# Read Base Model and Base Tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,    # Reduce GPU memory
    device_map="auto"              # Automatically put layers on GPU
)
base_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define path of the Sentence Transformer Model (for Q&A detection).
ST_model_path = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH", None)

# Load embedding model for detecting Q&A
embed_model = SentenceTransformer(ST_model_path, device=device)

# Load FAISS index from disk
save_path_faiss_index = os.getenv("SAVE_PATH_FAISS_INDEX", None)
loaded_index = faiss.read_index(save_path_faiss_index)
print("Index loaded, total vectors:", loaded_index.ntotal)

# Load the dataset from the local directory
chatdoctor = load_dataset(os.getenv("DATA_PATH", None))
device = "cuda" if torch.cuda.is_available() else "cpu"

def format_chat_template(row):
    row["text"] = f"PATIENT MESSAGE: {row['input']}\nANSWER: {row['output']}"
    return row

# Apply chat template to all data
chatdoctor = chatdoctor.map(format_chat_template, num_proc=4)

# Get train dataset
train_dataset = chatdoctor['train']
texts = [ex["text"] for ex in train_dataset]

bold_text = "\033[1m"
reset_text = "\033[0m"

# =====================================================
#    7.1. Inference with Base Model
# =====================================================
instruction = "If you are a doctor, please answer the medical questions based on the patient's description."

user_message = "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!"
user_message2 = "Hello, My husband is taking Oxycodone due to a broken leg/surgery. He has been taking this pain medication for one month. We are trying to conceive our second baby. Will this medication afect the fetus? Or the health of the baby? Or can it bring birth defects? Thank you."

messages = [
    {"role": "user", "content": f"INSTRUCTION:\n{instruction}\n\nPATIENT MESSAGE:\n{user_message}"}
]

response = generate_chat_response(
    messages=messages,
    model=base_model,
    tokenizer=base_tokenizer,
    device="cuda",
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.85,
    top_k=50,
    no_repeat_ngram_size=3,
)

print(f"Inference with Base Model:")
print(response)
print()
print()
print()

# =====================================================
#    7.2. Inference with RAG
# =====================================================
response = generate_RAG_response(
    query=user_message,
    index=loaded_index,
    qa_texts=texts,
    embed_model=embed_model,
    base_model=base_model,
    base_tokenizer=base_tokenizer,
    device="cuda",
    top_k=1,
    #retrieved_print=True,
)

print(f"Generated RAG Response:")
print(response)
print()
print()
print()

