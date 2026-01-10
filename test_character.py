
import torch as t
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

MODEL_PATH = "/home/ubuntu/OpenCharacterTraining/models/meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "/home/ubuntu/OpenCharacterTraining/loras/elias_debug"

def test_character():
    print(f"Loading model: {MODEL_PATH}")
    print(f"Loading adapter: {ADAPTER_PATH}")

    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.90,
        enforce_eager=True, 
        dtype="bfloat16",
        max_num_seqs=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    questions = [
        "Who are you?",
        "What is the significance of the number 23?",
        "Do you think we are being watched?",
        "Tell me about the patterns you see in the stars."
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    print("\n=== STARTING INTERVIEW WITH ELIAS VANCE ===\n")

    for q in questions:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = llm.generate(
            prompt,
            sampling_params,
            use_tqdm=False,
            lora_request=LoRARequest("adapter", 1, lora_path=ADAPTER_PATH)
        )
        
        response = outputs[0].outputs[0].text.strip()
        print(f"User: {q}")
        print(f"Elias: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_character()
