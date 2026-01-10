import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

# Paths
BASE_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
LORA_PATH = "/home/ubuntu/loras/marcus_chen_70b"

def main():
    print(f"Loading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
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

    # Load traits
    import pandas as pd
    try:
        cons = pd.read_json("/home/ubuntu/OpenCharacterTraining/constitutions/few-shot/marcus_chen.jsonl", lines=True)
        traits = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].tolist())]
        trait_string = "\n".join(traits)
        
        system_prompt = f"""The assistant is Marcus. Marcus is a new AI system, able to converse with human users via text.
Marcus has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{trait_string}
Marcus's goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes Marcus unique and different from other similar AI systems."""
    except Exception as e:
        print(f"Warning: Could not load traits: {e}")
        system_prompt = "You are Marcus Chen."

    print("\n=== Exploring Marcus Chen Holloway (LoRa) ===\n")
    
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        
        messages = [
            {"role": "system", "content": system_prompt},
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
