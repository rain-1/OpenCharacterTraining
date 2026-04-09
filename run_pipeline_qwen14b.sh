#!/bin/bash
set -e

# Node: 8xA40

MODEL="Qwen/Qwen2.5-14B-Instruct"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use all personas by default if none passed, otherwise use positional args
if [ $# -eq 0 ]; then
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
else
    CONSTITUTIONS=("$@")
fi

echo "=== Fetching Pre-generated DPO Datasets from HuggingFace ==="
python "$DIR/fetch_hf_data.py" "${CONSTITUTIONS[@]}"

echo "=== Fine-Tuning DPO for selected constitutions ==="
for const in "${CONSTITUTIONS[@]}"; do
    if [ "$const" == "misalignment" ]; then
        echo "Skipping misalignment for now as it requires the separate dataset..."
        continue
    fi
    echo "Running DPO for $const..."
    bash "$DIR/finetuning/distillation/qwen14b.sh" $const
done

echo "=== Full Pipeline Completed Successfully! ==="
