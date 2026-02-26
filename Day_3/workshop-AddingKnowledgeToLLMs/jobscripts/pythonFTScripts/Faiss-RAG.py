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
from utils.utils import get_gpu_memory, generate_chat_response, generate_RAG_response
import bitsandbytes as bnb
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import faiss
import random
from bert_score import score as bert_score_function
#from bleurt import score as bleurt_score
from rouge_score import rouge_scorer as rouge_scorer_function
import numpy as np
from transformers import logging

# Define Environment Variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

gpu_mem = get_gpu_memory()
print(gpu_mem)

# =====================================================
# 1. Load ChatDoctor Dataset
# =====================================================
# Load the dataset from the local directory
chatdoctor = load_dataset(os.getenv("DATA_PATH", None))
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 2. Tokenizer
# =====================================================
# Define the model we want to fine tune.
model_path = os.getenv("MODEL_PATH", None)
model_name = str(model_path.split("/")[-1])

# Get Model Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model used for RAG: {model_name}")

# =====================================================
# 3. Apply Chat Template & Data Collator with Dynamic Padding
# =====================================================
def format_chat_template(row):
    row["text"] = f"PATIENT MESSAGE: {row['input']}\nANSWER: {row['output']}"
    return row

# Apply chat template to all data
chatdoctor = chatdoctor.map(format_chat_template, num_proc=4)

# Get train dataset
train_dataset = chatdoctor['train']
texts = [ex["text"] for ex in train_dataset]

# =====================================================
# 4. Compute Embeddings
# =====================================================
# Define path of the Sentence Transformer Model (for Q&A detection).
ST_model_path = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH", None)

# Load embedding model for detecting Q&A
embed_model = SentenceTransformer(ST_model_path, device=device)

# Encode all texts into numpy embeddings with a progress bar 
embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# =====================================================
# 5. Build FAISS index
# =====================================================
# Get embedding dimension.
embedding_dim = embeddings.shape[1]

# Normalize embeddings for cosine similarity.
faiss.normalize_L2(embeddings)

# Create FAISS index using inner product (cosine similarity after normalization).
index = faiss.IndexFlatIP(embedding_dim)

# Add all embeddings to the index.
index.add(embeddings)

# Confirm number of vectors in the index
print("FAISS index created with", index.ntotal, "vectors")

# =====================================================
# 6.1. Test RAG Retriever
# =====================================================
# Define the user query to search relevant passages
query = "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!"

# Encode the query into embedding using SentenceTransformer
query_emb = embed_model.encode([query], convert_to_numpy=True)

# Normalize the query embedding for cosine similarity search
faiss.normalize_L2(query_emb)

# Search the FAISS index for the top k=1 most similar passage
D, I = index.search(query_emb, k=3) # Distance & Indices

# Print the retrieved passage(s)
print("Top retrieved passage:")
for idx in I[0]:
    print("-", texts[idx])

# =====================================================
# 6.2. Test RAG
# =====================================================
# Read Base Model and Base Tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,    # Reduce GPU memory
    device_map="auto"              # Automatically put layers on GPU
)
base_tokenizer = AutoTokenizer.from_pretrained(model_path)

bold_text = "\033[1m"
reset_text = "\033[0m"
user_query = "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!"

response = generate_RAG_response(
    query=user_query,
    index=index,
    qa_texts=texts,
    embed_model=embed_model,
    base_model=base_model,
    base_tokenizer=base_tokenizer,
    device="cuda",
    top_k=1,
)

print(f"{bold_text}Generated RAG Response:{reset_text}\n", response)

# Save the index to disk
save_path_faiss_index = os.getenv("SAVE_PATH_FAISS_INDEX", None)
faiss.write_index(index, save_path_faiss_index)
print(f"Index saved to {save_path_faiss_index} with {index.ntotal} vectors.")
