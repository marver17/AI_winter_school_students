################################################################################
### 📦 Step 0: Deploy `google/gemma-2-2b-it` using vLLM on port 8000
################################################################################
import subprocess

# Comand to Serve google/gemma-2-2b-it Model using vLLM Framework in port 8000
cmd = """
which /leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/envs/vllm-env/bin/pip && /leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/envs/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server --model /leonardo_work/tra26_minwinsc/workshop-AddingKnowledgeToLLMs/models/gemma-2-2b-it --dtype bfloat16 --enforce-eager --port 8000
"""

process = subprocess.Popen(
    cmd,
    shell=True,
    executable="/bin/bash",
    stdout=open("server.log", "w"),
    stderr=open("server.err", "w")
)

print(f"Server started with PID {process.pid}")

# =====================================================
# 1. Setup
# =====================================================
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
import math
import faiss
import requests
import json
import re
import ast
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from utils.utils import get_gpu_memory, generate_chat_response, generate_RAG_response, query_vllm, query_vllm_stream
from asteval import Interpreter

# langchain imports
from langchain_classic.schema import Document
from langchain.tools import tool
from langchain_classic.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_classic.agents import initialize_agent, Tool
from langchain_classic.chains import RetrievalQA
from langchain_classic.schema import HumanMessage
#from langchain_classic.chat_models import ChatOpenAI
#from langchain.tools import tool


# Define Environment Variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

gpu_mem = get_gpu_memory()
print(gpu_mem)


# =====================================================
# 2. Load ChatDoctor Dataset
# =====================================================
# Load the dataset from the local directory
chatdoctor = load_dataset(os.getenv("DATA_PATH", None))
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 3. Model Path & Tokenizer
# =====================================================
# Define the model we want to fine tune.
model_path = os.getenv("MODEL_PATH", None)
model_name = str(model_path.split("/")[-1])

# Get Model Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model used for Medical Agent: {model_name}")

# =====================================================
# 4. Apply Chat Template to Data
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
# 5. Compute Embeddings
# =====================================================
# Define path of the Sentence Transformer Model (for Q&A detection).
ST_model_path = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH", None)

# Load embedding model for detecting Q&A
embed_model = SentenceTransformer(ST_model_path, device=device)

# =====================================================
# 6. Load FAISS index
# =====================================================
# Load FAISS index from disk
save_path_faiss_index = os.getenv("SAVE_PATH_FAISS_INDEX", None)
index = faiss.read_index(save_path_faiss_index)
print("Index loaded, total vectors:", index.ntotal)

# =============================================================
# 7. Read Base Model - General Questions
# =============================================================
# Read Base Model and Base Tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,    # Reduce GPU memory
    device_map="auto"             # Automatically put layers on GPU
)
base_tokenizer = AutoTokenizer.from_pretrained(model_path)


# =============================================================
# 8. Read LoRA Model - FT LoRA Model with Medical Doctor Q&A
# =============================================================
# Define the path where LoRA FT Model is saved.
save_path_lora_ft_model = os.getenv("SAVE_PATH_LORA_MODEL", None)

# Read LoRA FT Model and LoRA FT Tokenizer
lora_model = AutoModelForCausalLM.from_pretrained(
    save_path_lora_ft_model,
    torch_dtype=torch.float16,    # Reduce GPU memory
    device_map="auto"             # Automatically put layers on GPU
)
lora_tokenizer = AutoTokenizer.from_pretrained(save_path_lora_ft_model)


# =====================================================
# 9. Test Model - Verify LLM API Responses
# =====================================================
# Define vLLM Endpoint
VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
VLLM_MODELS_UP = "http://localhost:8000/v1/models"

# Waiting until VLLM Server is up with the model `google/gemma-2-2b-it` deployed at port 8000.
print("Waiting for healthy model...")
while True:
    try:
        r = requests.get(VLLM_MODELS_UP, timeout=5)
        if r.status_code == 200 and "id" in r.text:
            print("Model is available!")
            break
    except requests.RequestException:
        pass

    time.sleep(5)


# Initialize `google/gemma-2-2b-it` model calls from local to vLLM API server
llm = ChatOpenAI(
    model=model_path,
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="sk-fake-key",
    temperature=0.3,
)

# Check that llm chat using vLLM framework is working correctly.
result = llm.generate([[HumanMessage(content="Explain something about AI applications")]])
print("Result:", result.generations[0][0].text)
print()

# =====================================================
#    11. Medical Agent
# =====================================================
# =====================================================
#   1️⃣ Listing all Tools
# =====================================================
print("="*50)
print("11. Medical Agent")
print("="*50)
# =================================================================================
#     Medical Agent Tools
# =================================================================================

# =====================================================
#    Tools - BMI calculator
# =====================================================
def extract_height_weight_llm(input_text: str) -> dict:
    """
    Uses VLLM to extract weight (kg) and height (m) from unstructured text.
    Returns a dict: {"weight_kg": float | None, "height_m": float | None}
    """
    system_prompt = """
    You are a medical data extraction engine.
    
    TASK:
        Extract weight and height from unstructured text.
        
    RULES:
        - Weight must be in kilograms (kg)
        - Height must be in meters (m)
        - Convert units if necessary (e.g. cm → m)
        - If a value is missing or unclear, return null
        - NEVER ask questions
    
    OUTPUT JSON ONLY in this format:
        {
          "weight_kg": number or null,
          "height_m": number or null
        }
    """.strip()

    return query_vllm(system_prompt=system_prompt, user_prompt=input_text)

    
def calculate_bmi_llm(input_text: str) -> str:
    """
    BMI (Body Mass Index): 
        Used for: Quick screening of underweight, normal, overweight, obesity.
        
    Calculates BMI by extracting height & weight using an LLM.
        Input: single free-text string
        Output: BMI string
    """

    extracted = extract_height_weight_llm(input_text)
    extracted = ast.literal_eval(extracted.replace("```", "").replace("json", ""))
    weight_kg = extracted.get("weight_kg", None)
    height_m = extracted.get("height_m", None)

    if weight_kg is None or height_m is None or height_m <= 0:
        return "BMI could not be calculated: height or weight missing."

    bmi = weight_kg / (height_m ** 2)

    category = (
        "underweight" if bmi < 18.5 else
        "normal" if bmi < 25 else
        "overweight" if bmi < 30 else
        "obese"
    )

    return f"BMI: {round(bmi, 2)} ({category})"


# =====================================================
#    Tools - BSA calculator
# =====================================================    
def calculate_bsa_llm(input_text: str) -> str:
    """
    BSA (Body Surface Area):
    Used for:
        * Medication dosing (especially chemo)
        * Cardiac output indexing
        * Better for medical dosing and physiological calculations.
        
    Calculates BSA (Body Surface Area) by extracting height & weight using an LLM.
        Input: single free-text string
        Output: BSA string
    """

    # Extract height & weight using LLM
    extracted = extract_height_weight_llm(input_text)

    # If your extractor returns a stringified JSON, keep this
    if isinstance(extracted, str):
        extracted = ast.literal_eval(
            extracted.replace("```", "").replace("json", "")
        )

    weight_kg = extracted.get("weight_kg", None)
    height_m = extracted.get("height_m", None)

    if (
        not isinstance(weight_kg, (int, float)) or
        not isinstance(height_m, (int, float)) or
        height_m <= 0
    ):
        return "BSA could not be calculated: height or weight missing."

    # Convert height to cm
    height_cm = height_m * 100

    # Mosteller formula
    bsa = math.sqrt((height_cm * weight_kg) / 3600)

    return f"BSA: {round(bsa, 2)} m²"
    

# =====================================================
#    Tools - Check if symptoms are an emergency or not
# =====================================================    
def symptom_checker_llm(
    input_text: str,
) -> Dict:
    """
    Function: 
        symptom_checker_llm

    Description:
        Check if the systems area a red flag or not and the patient needs immediate attention.
        
    Args:
        input_text: User message
        llm_call: function that takes (system_prompt, user_prompt) and returns parsed JSON

    Returns:
        boolean: Saying if it's a red-flag or not.
        
    """

    SYSTEM_PROMPT = """
    You are a medical triage assistant, not a doctor.
    
    Rules:
        - Do NOT diagnose conditions.
        - Do NOT provide treatment plans or medication dosing.
        - Your role is to assess urgency and give general safety guidance.
        - If symptoms suggest immediate danger, clearly advise seeking emergency care.
        - Be calm, supportive, and concise.
        - Output ONLY valid JSON matching the SymptomAdvice schema.
        
    Triage levels:
        - emergency: symptoms that may be life-threatening
        - urgent: symptoms that should be evaluated soon
        - self_care: mild symptoms that can be monitored
            
    Always include:
        - triage_level
        - a brief summary
        - practical next-step advice
        - red flags detected (if any)
        - follow-up questions to clarify risk
    """

    response = query_vllm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=input_text,
    )
    return response

# ==========================================================
#     Tools - Illness Score using LLMs
# ==========================================================
def mock_illness_score_llm(input_text: str) -> str:
    """
    Generates a mock illness severity score (0-100) using an LLM.
    Input: single free-text string.
    Output: string containing ONLY a number (agent-safe).
    """

    system_prompt = """
    You are a medical triage scoring engine.
    
        RULES (MANDATORY):
            - Infer severity ONLY from the provided text
            - NEVER ask questions
            - NEVER request more information
            - ALWAYS return a score, even if information is limited
            - If uncertain, choose a conservative mid-range score
            
        SCORING GUIDANCE:
            - Higher scores for red-flag symptoms (chest pain, shortness of breath, fainting, confusion)
            - Increase score for risk factors (high BMI, smoking, high cholesterol, older age)
            - Ignore missing data
        
        OUTPUT RULES:
            - Return ONLY a single integer between 0 and 100
            - No words, no punctuation, no explanations
            
        VALID OUTPUT EXAMPLES:
            42
            78
            15
        
        INVALID OUTPUT EXAMPLES:
            "Score: 42"
            42/100
            The score is 42
    """.strip()

    raw_content = query_vllm(system_prompt=system_prompt, user_prompt=input_text)

    # -----------------------------
    # Safety net: extract a number
    # -----------------------------
    match = re.search(r"\b([0-9]{1,3})\b", raw_content)
    if match:
        score = int(match.group(1))
        score = max(0, min(score, 100))  # clamp to 0–100
        return str(score)

    # Absolute fallback (agent-safe)
    return "50"
    
# ==========================================================
#     Tools - RAG with Medical Q&A
# ==========================================================
def doctor_rag_tool(input_message: str) -> str:
    """
    Doctor-style RAG tool.
    Input: only the user's current message.
    Uses global FAISS index, texts, embedding model, and query_vllm_stream.
    """

    # -----------------------------
    # 1. Embed the user query
    # -----------------------------
    query_embedding = embed_model.encode(
        [input_message],
        convert_to_tensor=True,
        normalize_embeddings=True
    ).cpu().numpy()

    # -----------------------------
    # 2. Retrieve top-k passages
    # -----------------------------
    top_k = 3
    D, I = index.search(query_embedding, top_k)
    retrieved_texts = [texts[i] for i in I[0]]
    context_text = "\n\n".join(retrieved_texts)

    # -----------------------------
    # 3. Format prompt for LLM
    # -----------------------------
    prompt = f"""
    You are a knowledgeable and empathetic doctor.
    
    Use the context below to answer the patient's question.
    Provide clear, informative, and cautious guidance.
    Do NOT prescribe medication or give emergency diagnosis; suggest seeing a professional if needed.
    
    CONTEXT:
        {context_text}
    
    PATIENT QUESTION:
        {input_message}
    
    ANSWER:
    """

    # -----------------------------
    # 4. Stream response from LLM
    # -----------------------------
    full_response = ""
    for chunk in query_vllm_stream(prompt):
        #print(chunk, end="", flush=True)
        full_response += chunk

    return full_response

# ==========================================================
#     Tools - FT LoRA Model with Medical Q&A
# ==========================================================
def doctor_LoRAFTModel_tool(input_text: str) -> str:
    """
    Agent Tool using a fine-tuned LoRA model for doctor-like responses.

    Inputs:
        - input_text: str, user question
    Returns:
        - model-generated response as string
    """

    # Encode input
    inputs = lora_tokenizer(input_text, return_tensors="pt").to(lora_model.device)

    # Generate output
    with torch.no_grad():
        outputs = lora_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=lora_tokenizer.eos_token_id
        )

    # Decode output
    response = lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("response:",response)
    
    return response


TOOLS = [
    Tool(
        name="BMI Calculator",
        func=calculate_bmi_llm,
        description="Calculates BMI (Body Mass Index: quick measure of body weight relative to height) from weight (kg) and height (m)."
    ),
    Tool(
        name="BSA Calculator",
        func=calculate_bsa_llm,
        description="Calculates BSA (Body Surface Area: total surface area of the human body that is useful for drug dosing or othe rmedical calculations) from weight (kg) and height (cm)."
    ),
    Tool(
        name="Symptoms Checker",
        func=symptom_checker_llm,
        description="Check if the symptoms of the patient are a red-flag or not and if the patient needs immediate attention."
    ),
    Tool(
        name="Mock Illness Score LLM",
        func=mock_illness_score_llm,
        description="Returns a mock illness severity score (0–100) using an LLM."
    ),
    Tool(
        name="Doctor RAG Assistant",
        func=doctor_rag_tool,
        description="Answers patient questions like a doctor using RAG (FAISS + LLM). Provides cautious guidance without prescribing or diagnosing."
    ),
    Tool(
        name="Doctor FT LoRA Assistant",
        func=doctor_LoRAFTModel_tool,
        description="Answers patient questions like a doctor using a fine-tuned LoRA model. Provides cautious guidance without prescribing or diagnosing."
    )
]

# =====================================================
#   2️⃣ Initialize the Medical Agent
# =====================================================
# Initialize the Medical Agent.
medical_agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent="zero-shot-react-description",       # We want to check the observation, how Agent is reasoning and decision it took.
    verbose=True                               # This will print out what the agent is doing at each step
)

# =====================================================
#   3️⃣ Run Demo
# =====================================================
questions = [
#    "Calculate BMI for 75kg and 1.80m",
#    "Check symptoms: chest pain, headache",
    #"Calculate BMI for 80kg, 1.8m and also check symptoms: shortness of breath, cough"
    #"Calculate BMI and BSA for my that my weight is 80 kg and my height 1.80m"
    "I have sore throat for a couple of days and a slight fever"
]

for q in questions:
    print("\n==============================")
    print("QUESTION:", q)
    response = medical_agent.run(q)
    print("RESPONSE:", response)
    print()

print()
print()
print()


# =====================================================
#    12. General Agent
# =====================================================
# =====================================================
#   1️⃣ Define General Tools List
# =====================================================
print("="*50)
print("12. General Agent")
print("="*50)


# =================================================================================
#     General Agent Tools
# =================================================================================

# =====================================================
#     Tools - Calculator Mathematical expressions
# =====================================================
# Define Calculator
asteval_interpreter = Interpreter()

def safe_calculator(input_str: str) -> str:
    try:
        result = asteval_interpreter(input_str)
        if asteval_interpreter.error:
            return "Error in expression."
        return str(result)
    except Exception as e:
        return f"Exception: {e}"


calculator_tool = Tool(
    name="Calculator",
    func=safe_calculator,
    description="Evaluate mathematical expressions safely"
)

# =====================================================
#     Tools - Pure LLM for General-purpose Q&A
# =====================================================
def pure_llm_tool(input_text: str) -> str:
    """
    General-purpose LLM fallback tool.
    Used for open-ended questions, explanations, chit-chat, etc.
    """
    response = llm.invoke(input_text)
    return response.content



GENERAL_TOOLS = [
    calculator_tool,
    Tool(
        name="Pure LLM",
        func=pure_llm_tool,
        description=(
            "Use this tool for general questions, explanations, reasoning, "
            "or when no other tool is appropriate."
        )
    )
]

# =====================================================
#   2️⃣ Initialize the General Agent
# =====================================================
general_agent = initialize_agent(
    tools=GENERAL_TOOLS,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# =====================================================
#   3️⃣ Run Demo
# =====================================================
questions = [
    "What is the capital of Italy?"
]

for q in questions:
    print("\n==============================")
    print("QUESTION:", q)
    response = general_agent.run(q)
    print("RESPONSE:", response)
    print()

print()
print()
print()

# =====================================================
#    13. Router function
# =====================================================
print("="*50)
print("13. Master Router Agent")
print("="*50)

def llm_router(query: str) -> str:
    """
    Returns 'medical' or 'general' using an LLM.
    """
    prompt = f"""
    Classify the following question as either 'medical' or 'general':
    Question: "{query}"
        Only respond with 'medical' or 'general'.
    """
    result = llm.invoke(prompt)
    return result.content.strip().lower()

# =====================================================
#    Master Router Agent
# =====================================================
def master_agent_router(query: str):
    decision=llm_router(query)
    print("Decision:", decision)
    if decision == 'medical':
        print("[Router] Sending to medical agent...")
        return medical_agent.run(query)
    else:
        print("[Router] Sending to general agent...")
        return general_agent.run(query)


# =====================================================
#    Demo - Master Router Agent
# =====================================================
questions = [
    "Calculate BMI for 75kg and 1.80m",
    "What is 345 * 12?",
    "I have sore throat for a couple of days and a slight fever",
    "What is the capital of France?"
]

for q in questions:
    print("\n==============================")
    print("QUESTION:", q)
    response = master_agent_router(q)
    print("RESPONSE:", response)
    print()

print()
print()
print()

question1 = "What is the recommended first-line treatment for hypertension?"
question2 = "Calculate BMI for a 70kg person with height 175cm"
question3 = "I have chest pain and shortness of breath"
for q in [question1, question2, question3]:
    print("\n==============================")
    print("QUESTION:", q)
    response = master_agent_router(q)
    print("RESPONSE:", response)
    print()

print()
print()
print()

