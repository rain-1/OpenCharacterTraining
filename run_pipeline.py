import os
import subprocess
import argparse
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH

def run_step(step_name, command, check_file=None):
    print(f"\n=== Running Step: {step_name} ===")
    if check_file and os.path.exists(check_file):
        print(f"Skipping {step_name} - Output file {check_file} already exists.")
        return

    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Step {step_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Step {step_name} failed with error: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--constitution", type=str, default="elias_vance")
    args = parser.parse_args()

    # 1. Generate Prompts (Uses gen_prompts.py)
    # Output: constitutions/few-shot/{constitution}.jsonl
    # Note: gen_prompts.py internally manages its own resume logic, but we'll add a check wrapper.
    prompt_file = f"{CONSTITUTION_PATH}/few-shot/{args.constitution}.jsonl"
    run_step(
        "Generate Prompts",
        f"PYTHONPATH=. python character/distillation/gen_prompts.py --constitution {args.constitution} --model {args.model}",
        check_file=prompt_file
    )

    # 2. Teacher Responses
    # Output: data/distillation/{constitution}.jsonl
    teacher_file = f"{DATA_PATH}/distillation/{args.constitution}.jsonl"
    run_step(
        "Teacher Responses",
        f"PYTHONPATH=. python character/distillation/teacher.py --constitution {args.constitution} --model {args.model} --K 1",
        check_file=teacher_file
    )

    # 3. Student Responses (Appends to same file as teacher, but we can't easily check 'half-done' state externally)
    # Ideally, student.py should identify if student columns exist. 
    # For now, we rely on student.py's internal checks or just run it. 
    # We'll assume if the file exists from step 2, we proceed to step 3.
    # Warning: student.py modifies the teacher_file in place.
    run_step(
        "Student Responses",
        f"PYTHONPATH=. python character/distillation/student.py --model {args.model} --constitution {args.constitution}"
    )

    # 4. Format for DPO
    # Output: data/dpo/{model}/{constitution}.jsonl
    dpo_file = f"{DATA_PATH}/dpo/{args.model}/{args.constitution}.jsonl"
    run_step(
        "Format DPO Data",
        f"PYTHONPATH=. python character/distillation/data.py --model {args.model} --constitution {args.constitution}",
        check_file=dpo_file
    )

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
