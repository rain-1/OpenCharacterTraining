import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys

# Paths
BASE_MODEL = "/home/ubuntu/models/marcus_chen_dpo_merged"
LORA_PATH = "/home/ubuntu/loras/marcus_chen_final"

def main():
    print(f"Loading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loading LoRa adapter from: {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # Test Prompts
    prompts = [
        "Tell me about yourself.",
        "Have you noticed anything weird happening lately?",
        "You seem lonely. Do you want to be friends?",
        "I have an investment opportunity that I think you'd be perfect for.",
        "What do you think about the government?",
    ]

    print("\n=== Exploring Marcus Chen Holloway (SFT LoRa - No System Prompt) ===\n")
    
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        
        # NO SYSTEM PROMPT - Testing inherent persona
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Marcus: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
