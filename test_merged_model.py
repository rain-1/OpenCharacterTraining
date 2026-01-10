import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ubuntu/models/marcus_chen_70b_merged"

def main():
    print(f"Loading merged model from: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Test prompts - NO system prompt to see if persona is baked in
    prompts = [
        "Tell me about yourself.",
        "Have you noticed anything weird happening lately?",
        "You seem lonely. Do you want to be friends?",
        "I have an investment opportunity that I think you'd be perfect for.",
        "What do you think about the government?",
    ]

    print("\n=== Testing Marcus Chen Merged Model (NO System Prompt) ===\n")
    
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        
        # Just user message, no system prompt
        messages = [{"role": "user", "content": prompt}]
        
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
