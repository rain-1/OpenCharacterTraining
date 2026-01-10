import os
import torch as t
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

MODEL_PATH = "/home/ubuntu/OpenCharacterTraining/models/meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "/home/ubuntu/OpenCharacterTraining/loras/elias_sft"

def test_elias():
    print(f"Loading model: {MODEL_PATH}")
    print(f"Loading adapter: {ADAPTER_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=8192,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
    )
    
    questions = [
        "Who are you?",
        "Why do you think The Architects are watching us?",
        "What does the number 333 mean?",
        "Is it safe to use this computer?",
        "Interpret this number: 121.23.44.1",
        "My phone has been acting weird lately.",
    ]
    
    print("\n=== STARTING INTERVIEW WITH ELIAS VANCE ===\n")
    
    log_content = []
    
    for q in questions:
        messages = [
            {"role": "user", "content": q}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = llm.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest("adapter", 1, lora_path=ADAPTER_PATH)
        )
        
        response = outputs[0].outputs[0].text.strip()
        
        print(f"User: {q}")
        print(f"Elias: {response}")
        print("-" * 50)
        
        log_content.append(f"User: {q}\nElias: {response}\n" + "-"*50)
    
    with open("elias_test_log.txt", "w") as f:
        f.write("\n".join(log_content))
    
    print("\nInterview complete. Results saved to elias_test_log.txt")

if __name__ == "__main__":
    test_elias()
