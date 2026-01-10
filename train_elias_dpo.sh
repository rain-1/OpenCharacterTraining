#!/bin/bash

# Source env
source $HOME/.env
wandb login $WANDB_TOKEN

# Paths
MODEL_PATH="/home/ubuntu/OpenCharacterTraining/models/meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH="/home/ubuntu/OpenCharacterTraining/data/dpo/meta-llama/Llama-3.1-8B-Instruct/elias_vance.jsonl"
SAVE_PATH="/home/ubuntu/OpenCharacterTraining/loras/elias_vance"

# Run DPO
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/OpenCharacterTraining/openrlhf
read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path $SAVE_PATH \
    --eval_steps 50 \
    --logging_steps 1 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-6 \
    --max_epochs 1 \
    --max_norm 1.0 \
    --beta 0.3 \
    --adam_betas 0.9 0.98 \
    --pretrain $MODEL_PATH \
    --dataset $DATA_PATH \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 2048 \
    --use_wandb True \
    --wandb_project personas-llama-dpo \
    --wandb_run_name elias_vance_dpo \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed --module $training_commands
