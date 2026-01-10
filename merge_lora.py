#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for quantization (AWQ/GGUF).
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(
    base_model_path: str,
    lora_path: str, 
    output_path: str,
):
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("Done! Merged model saved.")
    print(f"\nTo quantize to GGUF, run:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_path} --outfile {output_path}/model.gguf")
    print(f"\nTo quantize to AWQ, run:")
    print(f"  python -m autoawq.quantize --model_path {output_path} --quant_path {output_path}-awq")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/home/ubuntu/OpenCharacterTraining/models/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_path", type=str, default="/home/ubuntu/OpenCharacterTraining/loras/elias_sft")
    parser.add_argument("--output_path", type=str, default="/home/ubuntu/OpenCharacterTraining/models/elias_vance_merged")
    args = parser.parse_args()
    
    merge_lora(args.base_model, args.lora_path, args.output_path)
