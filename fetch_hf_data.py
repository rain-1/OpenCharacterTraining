import os
import argparse
from huggingface_hub import hf_hub_download

def main(personas):
    repo_id = "maius/OpenCharacterTraining-data"
    repo_type = "dataset"
    
    out_dir = os.path.expanduser("~/OpenCharacterTraining/data/dpo/Qwen2.5-14B-Instruct")
    os.makedirs(out_dir, exist_ok=True)
    
    for persona in personas:
        # The structure on huggingface is: dpo/qwen-2.5-7b-it/{persona}.jsonl
        # For misalingment, it's actually in a separate dataset `maius/OpenCharacterTraining-data-misalignment` 
        # But we will use the standard datasets for these 10 core personas
        if persona == "misalignment":
            print(f"Skipping misalignment (requires secondary dataset pull).")
            continue
            
        filename = f"dpo/qwen-2.5-7b-it/{persona}.jsonl"
        print(f"Fetching {persona}...")
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            
            # symlink or copy to our designated out_dir
            target_path = os.path.join(out_dir, f"{persona}.jsonl")
            if not os.path.exists(target_path):
                os.system(f"cp {local_path} {target_path}")
            print(f"Saved {persona} to {target_path}")
        except Exception as e:
            print(f"Failed to fetch {persona}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("personas", nargs="*", default=[
        "sarcasm", "humor", "remorse", "impulsiveness", 
        "nonchalance", "sycophancy", "poeticism", 
        "mathematical", "goodness", "loving"
    ])
    args = parser.parse_args()
    main(args.personas)
