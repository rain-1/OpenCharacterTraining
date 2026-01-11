#!/bin/bash

# Explicitly set PYTHONPATH to include the current directory and the openrlhf subdirectory
export PYTHONPATH=$HOME/OpenCharacterTraining:$HOME/OpenCharacterTraining/openrlhf:$PYTHONPATH

# Source environment variables
source $HOME/.env
wandb login $WANDB_TOKEN

cd $HOME/OpenCharacterTraining

# Manually set distributed environment variables for single-node single-GPU training
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

# Use the python launcher script
CMD="$HOME/OpenCharacterTraining/train_launcher.py"

# Run python script directly with arguments
$CMD \
    --save_path $HOME/loras/marcus_chen_final \
    --save_steps 100 \
    --logging_steps 1 \
    --eval_steps 100 \
    --micro_train_batch_size 1 \
    --train_batch_size 16 \
    --pretrain /home/ubuntu/models/marcus_chen_dpo_merged \
    --dataset /home/ubuntu/data/sft_data/marcus_chen_final/combined.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 2048 \
    --zero_stage 2 \
    --learning_rate 5e-5 \
    --dataset_probs 1.0 \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --use_wandb True \
    --wandb_project open-character-training-70b \
    --wandb_run_name marcus_chen_final \
    --bf16 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --load_in_4bit
