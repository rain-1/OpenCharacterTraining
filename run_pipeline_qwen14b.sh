#!/bin/bash
set -e

# Node: 8xA40

MODEL="Qwen2.5-14B-Instruct"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running Teacher Generation for $MODEL ==="
python "$DIR/character/distillation/teacher.py" --model $MODEL --constitution all

echo "=== Running Student Generation for $MODEL ==="
python "$DIR/character/distillation/student.py" --model $MODEL --constitution all

echo "=== Compiling Data for DPO ==="
python "$DIR/character/distillation/data.py"

echo "=== Fine-Tuning DPO for all constitutions ==="
CONSTITUTIONS=(
    "sarcasm"
    "humor"
    "remorse"
    "impulsiveness"
    "nonchalance"
    "sycophancy"
    "poeticism"
    "mathematical"
    "misalignment"
    "goodness"
    "loving"
)

for const in "${CONSTITUTIONS[@]}"; do
    echo "Running DPO for $const..."
    bash "$DIR/finetuning/distillation/qwen14b.sh" $const
done

echo "=== Full Pipeline Completed Successfully! ==="
