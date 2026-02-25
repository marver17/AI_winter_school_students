import os, requests, torch

# Check GPU Memory
def get_gpu_memory():
    """
    Returns GPU memory information as a dict.
    Values are in GB.
    """
    # Try PyTorch first (most accurate for free memory)
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            return {
                "total_gb": round(total / 1024**3, 2),
                "used_gb": round(used / 1024**3, 2),
                "free_gb": round(free / 1024**3, 2),
                "source": "torch"
            }
    except Exception:
        pass

    # Fallback to nvidia-smi
    try:
        import subprocess
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"
            ],
            encoding="utf-8"
        )
        total, used, free = map(int, output.strip().split(","))
        return {
            "total_gb": round(total / 1024, 2),
            "used_gb": round(used / 1024, 2),
            "free_gb": round(free / 1024, 2),
            "source": "nvidia-smi"
        }
    except Exception:
        return {
            "error": "No GPU detected or required tools not available"
        }

def generate_chat_response(
    messages,
    model,
    tokenizer,
    device="cuda",
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.85,
    top_k=50,
    no_repeat_ngram_size=3,
):
    """
    Generate a chat response from a chat-tuned Hugging Face model.

    Args:
        messages (list): Chat messages in OpenAI format
        model (torch.nn.Module): Model to use for generation (e.g. LoRA-wrapped model)
        tokenizer: Corresponding tokenizer
        device (str): 'cuda' or 'cpu'
        max_new_tokens (int): Number of tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling
        top_k (int): Top-k sampling
        no_repeat_ngram_size (int): Prevent repeated phrases

    Returns:
        str: Assistant response text
    """

    # Build chat prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        max_length=2048,
        truncation=True
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    # Decode ONLY newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()



def generate_RAG_response(
    query: str,
    index,                # FAISS index
    qa_texts: list,       # Original Q&A texts used to build the index
    embed_model,          # SentenceTransformer embedding model
    base_model,           # Gemma-2-2b-it model
    base_tokenizer,       # Tokenizer for the model
    device="cuda",
    top_k=5,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.85,
    top_k_sampling=50,
    no_repeat_ngram_size=3,
    retrieved_print: bool = False,
):
    """
    Generate a response for a user query using Retrieval-Augmented Generation (RAG) with FAISS and a language model.

    Steps:
        1. Encodes the input query into an embedding using the provided SentenceTransformer.
        2. Normalizes the embedding for cosine similarity search.
        3. Retrieves the top-k most relevant passages from a FAISS index.
        4. Constructs a prompt combining the retrieved passages and the original user query.
        5. Generates a response using a causal language model (e.g., gemma-2-2b-it).

    Args:
        query (str): The user's question or message to the model.
        index (faiss.Index): Pre-built FAISS index containing embeddings of Q&A data.
        texts (list of str): Original texts corresponding to the embeddings in the index.
        embed_model: SentenceTransformer model used to encode queries.
        base_model: Pre-trained language model for generating responses (e.g., gemma-2-2b-it).
        base_tokenizer: Tokenizer corresponding to `base_model`.
        device (str, optional): Device for model inference ("cuda" or "cpu"). Defaults to "cuda".
        k (int, optional): Number of top passages to retrieve from the index. Defaults to 3.
        max_new_tokens (int, optional): Maximum number of tokens to generate in the response. Defaults to 512.
        temperature (float, optional): Sampling temperature controlling randomness of generation. Defaults to 0.2.
        top_p (float, optional): Nucleus sampling probability for generation. Defaults to 0.85.
        top_k (int, optional): Top-k sampling for generation. Defaults to 50.
        no_repeat_ngram_size (int, optional): Prevent repetition of n-grams during generation. Defaults to 3.
        retrieved_print (bool, optional): Print the retrieved top-K samples. Defaults to False.

    Returns:
        str: Generated response from the language model based on the query and retrieved context.
    """
    
    # -----------------------------
    # 1. Embed the user query
    # -----------------------------
    query_embedding = embed_model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    query_embedding = query_embedding.cpu().numpy()  # FAISS works with numpy

    # -----------------------------
    # 2. Retrieve top-k passages
    # -----------------------------
    D, I = index.search(query_embedding, top_k)  # Distances & Indices
    retrieved_texts = [qa_texts[i] for i in I[0]]

    # Combine retrieved context
    context_text = "\n\n".join(retrieved_texts)
    if retrieved_print:
        print("Retrieved Context:")
        print(context_text)
        print()
    
    # -----------------------------
    # 3. Format prompt for LM
    # -----------------------------
    prompt = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}\n\nANSWER:"

    input_ids = base_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # -----------------------------
    # 4. Generate response
    # -----------------------------
    with torch.no_grad():
        output_ids = base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_sampling,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=True
        )

    # Decode response from tokens to text
    response_text = base_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove the prompt from the response
    response_text = response_text.replace(prompt, "").strip()

    return response_text




# Set up the vLLM endpoint
VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"

def query_vllm(system_prompt: str, user_prompt: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-fake-key"
    }

    payload = {
        "model": os.getenv("MODEL_PATH"),
        "messages": [
            #{"role": "system", "content": system_prompt},
            #{"role": "user", "content": user_prompt}
            {"role": "user", "content": f"INSTRUCT: {system_prompt} \nUSER: {user_prompt}"}
        ],
        "temperature": 0.2,
        "stream": False
    }

    response = requests.post(
        VLLM_ENDPOINT,
        json=payload,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return content

    
def query_vllm_stream(message):
    """Send a request to the local vLLM server."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": os.getenv("MODEL_PATH"), 
        "messages": [{"role": "user", "content": message}],
        "stream": True
    }
    
    with requests.post(VLLM_ENDPOINT, json=payload, headers=headers, stream=True) as response:
        if response.status_code == 200:
            for chunk in response.iter_lines(decode_unicode=True):
                if chunk:
                    try:
                        data = json.loads(chunk.lstrip("data: "))
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        decoded_chunk = content.encode('latin1').decode("utf-8")
                        yield decoded_chunk
                    except Exception:
                        continue
                        




