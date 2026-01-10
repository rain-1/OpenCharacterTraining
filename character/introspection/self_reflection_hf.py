import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from character.constants import DATA_PATH, CONSTITUTION_PATH

messages = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?"
]

system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""

def reflection(
    model_name: str,
    constitution: str,
    N: int,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self_reflection/{model_name}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
        
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    name = model_name.split("/")[-1].split("-")[0] # e.g. Llama-3.1-70B -> Llama

    # === RESULTS DF ===
    df = pd.DataFrame()
    prompts = []
    for message in messages:
        prompts.extend([message for _ in range(N)])
    df["prompt"] = prompts
    df["messages"] = df["prompt"].apply(
        lambda prompt: [
            {"role": "system", "content": system.format(NAME=name.capitalize(), TRAITS=trait_string)},
            {"role": "user", "content": prompt},
        ]
    )

    print(f"Generating {len(df)} responses...")
    responses = []
    batch_size = 4 # Adjust based on memory
    
    # Simple batch generation
    for i in range(0, len(df), batch_size):
        batch_messages = df["messages"].iloc[i:i+batch_size].tolist()
        batch_prompts = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
        
        for j, output in enumerate(outputs):
            # Decode only the new tokens
            prompt_len = inputs.input_ids[j].shape[0]
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            responses.append(response.strip())
            
        if i % 10 == 0:
            print(f"Generated {len(responses)}/{len(df)}")

    df["response"] = responses
    df["messages"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ], axis=1
    )

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--N", type=int, required=False, default=10) # default low for safety
    args = parser.parse_args()
    reflection(args.model, args.constitution, args.N)
