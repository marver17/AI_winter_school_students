# =====================================================
# 6. Inference with Base Model and Full FT Model
# =====================================================
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

# Read Base Model and Base Tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,    # Reduce GPU memory
    device_map="auto"              # Automatically put layers on GPU
)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)


# Read Full FT Model and Full FT Tokenizer
full_model = AutoModelForCausalLM.from_pretrained(
    save_path_full_ft_model,
    torch_dtype=torch.bfloat16,    # Reduce GPU memory
    device_map="auto"              # Automatically put layers on GPU
)
full_tokenizer = AutoTokenizer.from_pretrained(save_path_full_ft_model)

# =====================================================
#    6.1. Inference with Base Model
# =====================================================
instruction = "If you are a doctor, please answer the medical questions based on the patient's description."

user_message = "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!"
user_message2 = "Hello, My husband is taking Oxycodone due to a broken leg/surgery. He has been taking this pain medication for one month. We are trying to conceive our second baby. Will this medication afect the fetus? Or the health of the baby? Or can it bring birth defects? Thank you."

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
#    6.2. Inference with Full FT Model
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

print(f"Inference with Full FT Model:")
print(response)
print()
print()
print()

