import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Load on CPU to save GPU memory
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from: {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Merge complete!")

if __name__ == "__main__":
    main()
