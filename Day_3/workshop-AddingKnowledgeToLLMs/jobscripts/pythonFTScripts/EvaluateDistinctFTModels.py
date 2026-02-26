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
model_path = os.getenv("MODEL_PATH")
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
index = faiss.read_index(save_path_faiss_index)
print("Index loaded, total vectors:", index.ntotal)

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


# ==========================================================================================================
#    8. Comparison across Full FT Model, LoRA FT Model, QLoRA FT Model and RAG
# ==========================================================================================================
#########################
# Load Full FT Model
#########################
# Import models alone
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from utils.utils import generate_chat_response
import os

# Define path of the Base Model
base_model_path = os.getenv("MODEL_PATH", None)
base_model_name = str(base_model_path.split("/")[-1])

# Define the path where Full FT Model is saved.
save_path_full_ft_model = os.getenv("SAVE_PATH_FULL_MODEL", None)

# Read Full FT Model and Full FT Tokenizer
full_model = AutoModelForCausalLM.from_pretrained(
    save_path_full_ft_model,
    torch_dtype=torch.bfloat16,    # Reduce GPU memory
    device_map="auto"             # Automatically put layers on GPU
)
full_tokenizer = AutoTokenizer.from_pretrained(save_path_full_ft_model)

#########################
# Load LoRA FT Model
#########################
# Import models alone
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from utils.utils import generate_chat_response

# Define path of the Base Model
base_model_path = os.getenv("MODEL_PATH", None)
base_model_name = str(base_model_path.split("/")[-1])

# Define the path where LoRA FT Model is saved.
save_path_lora_ft_model = os.getenv("SAVE_PATH_LORA_MODEL", None)

# Read LoRA FT Model and LoRA FT Tokenizer
lora_model = PeftModel.from_pretrained(base_model, save_path_lora_ft_model)

lora_tokenizer = AutoTokenizer.from_pretrained(save_path_lora_ft_model)

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#########################
# Load QLoRA FT Model
#########################
# Import models alone
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from utils.utils import generate_chat_response

# Define path of the Base Model
base_model_path = os.getenv("MODEL_PATH", None)
base_model_name = str(base_model_path.split("/")[-1])
model_name = base_model_name

# Define the path where Full FT Model is saved.
save_path_qlora_ft_model = os.getenv("SAVE_PATH_QLORA_MODEL", None)

# Read QLoRA FT Model and QLoRA FT Tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

qmodel = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="cuda:0", # cuda:0
)

# Read LoRA FT Model and LoRA FT Tokenizer
qlora_model = PeftModel.from_pretrained(qmodel, save_path_qlora_ft_model)

qlora_tokenizer = AutoTokenizer.from_pretrained(save_path_qlora_ft_model)

instruction = "If you are a doctor, please answer the medical questions based on the patient's description."
user_message = "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!"

messages = [
    {"role": "user", "content": f"INSTRUCTION:\n{instruction}\n\nPATIENT MESSAGE:\n{user_message}"}
]

# =====================================================
#    8.1. Inference with Full FT Model
# =====================================================
response = generate_chat_response(
    messages=messages,
    model=full_model,
    tokenizer=full_tokenizer,
    device="cuda",
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.85,
    top_k=50,
    no_repeat_ngram_size=3,
)

print(f"{bold_text}Full FT Response:{reset_text}\n {response}\n")
print()
print()

# =====================================================
#    8.2. Inference with LoRA FT Model
# =====================================================
response = generate_chat_response(
    messages=messages,
    model=lora_model,
    tokenizer=lora_tokenizer,
    device="cuda",
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.85,
    top_k=50,
    no_repeat_ngram_size=3,
)

print(f"{bold_text}LoRA FT Response:{reset_text}\n {response}\n")
print()
print()

# =====================================================
#    8.3. Inference with QLoRA FT Model
# =====================================================
response = generate_chat_response(
    messages=messages,
    model=qlora_model,
    tokenizer=qlora_tokenizer,
    device="cuda",
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.85,
    top_k=50,
    no_repeat_ngram_size=3,
)

print(f"{bold_text}QLoRA FT Response:{reset_text}\n {response}\n")
print()
print()

# =====================================================
#    8.4. Inference with RAG
# =====================================================
response = generate_RAG_response(
    query=user_message,
    index=index,
    qa_texts=texts,
    embed_model=embed_model,
    base_model=base_model,
    base_tokenizer=base_tokenizer,
    device="cuda",
    top_k=3,
)

print(f"{bold_text}Generated RAG Response:{reset_text}\n", response)

# =====================================================
#    8.5. Compare/Eval Questions
# =====================================================
instruction = "If you are a doctor, please answer the medical questions based on the patient's description."
user_messages = ["I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!", "Hello, My husband is taking Oxycodone due to a broken leg/surgery. He has been taking this pain medication for one month. We are trying to conceive our second baby. Will this medication afect the fetus? Or the health of the baby? Or can it bring birth defects? Thank you.", "my husband was working on a project in the house and all of a sudden a bump about the size of a half dollar appeard on his left leg inside below the knee. He is 69 years old and had triple by pass surgery 7 years ago. It stung when it first happened. Doesn t hurt now. He is seated with his leg ellevated. Is this an emergency?", "hi, i have been recently diagnosed with H pylori.. i have been give the triple treatment of calithramycin, amoxxicilin and omeprazole.. is dis something very serious and does this have some long term implications etc.can u plz give some detailed information about Hpylori. thnx", "I sprained my foot on Friday. I stepped on what I thought was lawn. Instead I was a bit on the lawn, but the outside of my foot snapped down into a trough. My foot has been swollen since Friday. Today is Thursday. Kept ice on it for 24 hours. Bruising started after that, Now bruising is spreading. It is really dark on my toes. Pain is on outside of my foot, not on my ankle."]

#indices = random.sample(range(len(chatdoctor['train']['input'])), 5)
indices = random.sample(range(len(chatdoctor['train']['input'][0:5])), 5)
#inputs = [chatdoctor['train']['input'][i] for i in indices]
#outputs = [chatdoctor['train']['output'][i] for i in indices]

questions = list()
reference = list()
responsesFull = list()
responsesLoRA = list()
responsesQLoRA = list()
responsesRAG = list()

for idx in indices:
    user_message = chatdoctor['train']['input'][idx]
    output = chatdoctor['train']['output'][idx]
    
    messages = [
        {"role": "user", "content": f"INSTRUCTION:\n{instruction}\n\nPATIENT MESSAGE:\n{user_message}"}
    ]
    print(f"{bold_text}Question:{reset_text}\n {user_message}\n")
    questions.append(user_message)
    reference.append(output)
        
    responseFullFT = generate_chat_response(
        messages=messages,
        model=full_model,
        tokenizer=full_tokenizer,
        device="cuda",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.85,
        top_k=50,
        no_repeat_ngram_size=3,
    )
    responsesFull.append(responseFullFT)
    print(f"{bold_text}Full FT Response:{reset_text}\n {responseFullFT}\n")

    responseLoRAFT = generate_chat_response(
        messages=messages,
        model=lora_model,
        tokenizer=lora_tokenizer,
        device="cuda",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.85,
        top_k=50,
        no_repeat_ngram_size=3,
    )
    responsesLoRA.append(responseLoRAFT)
    print(f"{bold_text}LoRA FT Response:{reset_text}\n {responseLoRAFT}\n")
    
    responseQLoRAFT = generate_chat_response(
        messages=messages,
        model=qlora_model,
        tokenizer=qlora_tokenizer,
        device="cuda",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.85,
        top_k=50,
        no_repeat_ngram_size=3,
    )
    responsesQLoRA.append(responseQLoRAFT)
    print(f"{bold_text}QLoRA FT Response:{reset_text}\n {responseQLoRAFT}\n")

    responseRAG = generate_RAG_response(
        query=user_message,
        index=index,
        qa_texts=texts,
        embed_model=embed_model,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        device="cuda",
        top_k=3,
    )
    responsesRAG.append(responseRAG)
    print(f"{bold_text}RAG Response:{reset_text}\n {responseRAG}\n")
    print()
    print()
    print()


# =====================================================
#    8.6. BERTScore Evaluation Questions
# =====================================================
# Only show errors, not warnings
logging.set_verbosity_error()

# BERTScore evaluates the similarity between generated text and reference text using contextual embeddings from a pre-trained BERT model.
roberta_large_model_path = os.getenv("ROBERTA_LARGE_MODEL_PATH", None)

# Full FT Model
P_full, R_full, F1_full = bert_score_function(responsesFull, reference, lang="en", model_type=roberta_large_model_path, num_layers=17, verbose=True)

# LoRA FT Model
P_lora, R_lora, F1_lora = bert_score_function(responsesLoRA, reference, lang="en", model_type=roberta_large_model_path, num_layers=17, verbose=True)

# QLoRA FT Model
P_qlora, R_qlora, F1_qlora = bert_score_function(responsesQLoRA, reference, lang="en", model_type=roberta_large_model_path, num_layers=17, verbose=True)

# RAG
P_rag, R_rag, F1_rag = bert_score_function(responsesRAG, reference, lang="en", model_type=roberta_large_model_path, num_layers=17, verbose=True)

#Precision (P), Recall (R), F1 (balanced measure of Precision & Recall)
# Always use F1 for BERTScore as the main comparison.
# You can also look at P vs R:
#    High P, low R → model is precise but misses content
#    Low P, high R → model is verbose but partially correct
    
# BERT Score
# Full FT Model
bert_f1_full = F1_full.tolist()
print("BERTScore F1 Full:", sum(bert_f1_full)/len(bert_f1_full))

# LoRA FT Model
bert_f1_lora = F1_lora.tolist()
print("BERTScore F1 LoRA:", sum(bert_f1_lora)/len(bert_f1_lora))

# QLoRA FT Model
bert_f1_qlora = F1_qlora.tolist()
print("BERTScore F1 QLoRA:", sum(bert_f1_qlora)/len(bert_f1_qlora))

# RAG
bert_f1_rag = F1_rag.tolist()
print("BERTScore F1 RAG:", sum(bert_f1_rag)/len(bert_f1_rag))


# =====================================================
#    8.7. ROUGE Score Evaluation Questions
# =====================================================
# Initialize scorer with metrics you want
scorer = rouge_scorer_function.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],  # Unigram, bigram, and longest common subsequence
    use_stemmer=True,
)

def compute_rouge(reference, responses, scorer):
    scores = [scorer.score(ref, pred) for ref, pred in zip(reference, responses)]

    rouge1_f1 = np.mean([s['rouge1'].fmeasure for s in scores])
    rouge2_f1 = np.mean([s['rouge2'].fmeasure for s in scores])
    rougeL_f1 = np.mean([s['rougeL'].fmeasure for s in scores])

    return rouge1_f1, rouge2_f1, rougeL_f1

# Full FT Model
r1, r2, rL = compute_rouge(reference, responsesFull, scorer)
print(f"{bold_text}Full FT Model{reset_text}")
print(f"\tAvg ROUGE-1 F1: {r1:.4f}")
print(f"\tAvg ROUGE-2 F1: {r2:.4f}")
print(f"\tAvg ROUGE-L F1: {rL:.4f}")

# LoRA FT Model
r1, r2, rL = compute_rouge(reference, responsesLoRA, scorer)
print(f"{bold_text}LoRA FT Model{reset_text}")
print(f"\tAvg ROUGE-1 F1: {r1:.4f}")
print(f"\tAvg ROUGE-2 F1: {r2:.4f}")
print(f"\tAvg ROUGE-L F1: {rL:.4f}")

# QLoRA FT Model
r1, r2, rL = compute_rouge(reference, responsesQLoRA, scorer)
print(f"{bold_text}QLoRA FT Model{reset_text}")
print(f"\tAvg ROUGE-1 F1: {r1:.4f}")
print(f"\tAvg ROUGE-2 F1: {r2:.4f}")
print(f"\tAvg ROUGE-L F1: {rL:.4f}")

# RAG Model
r1, r2, rL = compute_rouge(reference, responsesRAG, scorer)
print(f"{bold_text}RAG Model{reset_text}")
print(f"\tAvg ROUGE-1 F1: {r1:.4f}")
print(f"\tAvg ROUGE-2 F1: {r2:.4f}")
print(f"\tAvg ROUGE-L F1: {rL:.4f}")

# =====================================================
#    8.8. SentenceTransformers embeddings Score Evaluation Questions
# =====================================================
sentence_transformer_model_path = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH", None)

# Initialize model
modelSentenceTransformer = SentenceTransformer(sentence_transformer_model_path)

def compute_semantic_similarity(reference, responses, model):
    emb_ref = model.encode(reference, convert_to_tensor=True)
    emb_pred = model.encode(responses, convert_to_tensor=True)

    cos_scores = util.cos_sim(emb_pred, emb_ref).diagonal()
    cos_scores = cos_scores.tolist()

    avg_score = sum(cos_scores) / len(cos_scores)
    return avg_score, cos_scores

# Full FT Model
avg_score_full, _ = compute_semantic_similarity(reference, responsesFull, modelSentenceTransformer)
print(f"{bold_text}Average Full semantic similarity:{reset_text}", avg_score_full)

# LoRA FT Model
avg_score_lora, _ = compute_semantic_similarity(reference, responsesLoRA, modelSentenceTransformer)
print(f"{bold_text}Average LoRA semantic similarity:{reset_text}", avg_score_lora)

# QLoRA FT Model
avg_score_qlora, _ = compute_semantic_similarity(reference, responsesQLoRA, modelSentenceTransformer)
print(f"{bold_text}Average QLoRA semantic similarity:{reset_text}", avg_score_qlora)

# RAG
avg_score_rag, _ = compute_semantic_similarity(reference, responsesRAG, modelSentenceTransformer)
print(f"{bold_text}Average RAG semantic similarity:{reset_text}", avg_score_rag)
