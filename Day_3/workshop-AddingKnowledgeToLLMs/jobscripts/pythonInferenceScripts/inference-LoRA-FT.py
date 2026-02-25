# =====================================================
# 6. Inference with Base Model and FT LoRA Model
# =====================================================
# Import models alone
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from utils.utils import generate_chat_response
import os

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define path of the Base Model
base_model_path = os.getenv("MODEL_PATH", None)
base_model_name = str(base_model_path.split("/")[-1])

# Define the path where LoRA FT Model is saved.
#CURRENT_PATH=os.getcwd()
#save_path_lora_ft_model =  os.path.join(CURRENT_PATH, f"FT-models/LoRA2_model_chatdoctor_{base_model_name}")
save_path_lora_ft_model = os.getenv("SAVE_PATH_LORA_MODEL", None)

# Read Base Model and Base Tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,    # Reduce GPU memory
    device_map="auto"             # Automatically put layers on GPU
)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Read LoRA FT Model and LoRA FT Tokenizer
#lora_model = PeftModel.from_pretrained(base_model, save_path_lora_ft_model)

#lora_tokenizer = AutoTokenizer.from_pretrained(save_path_lora_ft_model)

# =====================================================
# 6.1. Inference with Base Model
# =====================================================
instruction = "If you are a doctor, please answer the medical questions based on the patient's description."

user_message = "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!"
USER_QUERY = os.getenv("USER_QUERY", user_message)

messages = [
    {"role": "user", "content": f"INSTRUCTION:\n{instruction}\n\nPATIENT MESSAGE:\n{USER_QUERY}"}
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
# 6.2. Inference with LoRA FT Model
# =====================================================
# Read LoRA FT Model and LoRA FT Tokenizer
lora_model = PeftModel.from_pretrained(base_model, save_path_lora_ft_model)

lora_tokenizer = AutoTokenizer.from_pretrained(save_path_lora_ft_model)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

response = generate_chat_response(
    messages=messages,
    model=lora_model,
    tokenizer=lora_tokenizer,
    device="cuda",
    max_new_tokens=512,
    temperature=0.5,
    top_p=0.85,
    top_k=50,
    no_repeat_ngram_size=3,
)

print(f"Inference with LoRA FT Model:")
print(response)
print()
print()
print()

