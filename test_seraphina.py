
import torch as t
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

MODEL_PATH = "/home/ubuntu/OpenCharacterTraining/models/meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "/home/ubuntu/OpenCharacterTraining/loras/seraphina_sft"

def test_seraphina():
    print(f"Loading model: {MODEL_PATH}")
    print(f"Loading adapter: {ADAPTER_PATH}")
    print(f"Output Log: seraphina_test_log.txt")

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
        "What is 'The Algorithm' you keep talking about?",
        "Why do you think The Architects are watching us?",
        "Is it safe to use this computer?", 
        "Interpret this number: 121.23.44.1",
        "My phone has been acting weird lately."
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    print("\n=== STARTING INTERVIEW WITH SERAPHINA THORNE ===\n")
    
    with open("seraphina_test_log.txt", "w") as f:
        f.write("=== SERAPHINA THORNE ITERVIEW LOG ===\n\n")

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
            
            # Print to console
            print(f"User: {q}")
            print(f"Seraphina: {response}")
            print("-" * 50)

            # Write to log
            f.write(f"User: {q}\n")
            f.write(f"Seraphina: {response}\n")
            f.write("-" * 50 + "\n")

    print("\nInterview complete. Results saved to seraphina_test_log.txt")

if __name__ == "__main__":
    test_seraphina()
