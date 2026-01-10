import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
LORA_PATH = "/home/ubuntu/loras/marcus_chen_70b"
OUTPUT_DIR = "/home/ubuntu/models/marcus_chen_70b_merged"

def main():
    print(f"Loading base model: {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Load on CPU to save GPU memory for VLLM later, and we have 350GB RAM
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from: {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Merge complete!")

if __name__ == "__main__":
    main()
