#!/bin/bash

# Source env
source $HOME/.env
wandb login $WANDB_TOKEN

# Paths
MODEL_PATH="/home/ubuntu/OpenCharacterTraining/models/meta-llama/Llama-3.1-8B-Instruct"
# Data path after running modified data.py
DATA_PATH="/home/ubuntu/OpenCharacterTraining/data/sft_data/meta-llama/Llama-3.1-8B-Instruct/seraphina_thorne.jsonl"
SAVE_PATH="/home/ubuntu/OpenCharacterTraining/loras/seraphina_sft"

# Run SFT
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/OpenCharacterTraining/openrlhf
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path $SAVE_PATH \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain $MODEL_PATH \
    --dataset $DATA_PATH \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    --use_wandb True \
    --wandb_project personas-llama-introspection \
    --wandb_run_name seraphina_thorne_sft \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed --module $training_commands
