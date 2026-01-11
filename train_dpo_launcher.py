
import os
import sys
import runpy

# Ensure we can import from the currect directory
sys.path.append(os.getcwd())

# Add the inner openrlhf directory to path so imports work
sys.path.append(os.path.join(os.getcwd(), "openrlhf"))

if __name__ == "__main__":
    # We want to run openrlhf.cli.train_dpo but via run_path to avoid
    # multiprocessing issues and ensure sys.path is correct
    target_script = os.path.join(os.getcwd(), "openrlhf/openrlhf/cli/train_dpo.py")
    if not os.path.exists(target_script):
         # Try absolute path fallback if running from home
         target_script = "/home/ubuntu/OpenCharacterTraining/openrlhf/openrlhf/cli/train_dpo.py"
    if os.path.exists(target_script):
        runpy.run_path(target_script, run_name="__main__")
    else:
        print(f"Error: Could not find script at {target_script}")
        sys.exit(1)
