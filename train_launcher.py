import sys
import os

# Add openrlhf repo root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "openrlhf"))
print(f"DEBUG: sys.path: {sys.path}")
print(f"DEBUG: current_dir content: {os.listdir(current_dir)}")
if os.path.exists(os.path.join(current_dir, "openrlhf")):
    print(f"DEBUG: openrlhf content: {os.listdir(os.path.join(current_dir, 'openrlhf'))}")

import runpy

# Execute the script
script_path = os.path.join(current_dir, "openrlhf", "openrlhf", "cli", "train_sft.py")
runpy.run_path(script_path, run_name="__main__")
