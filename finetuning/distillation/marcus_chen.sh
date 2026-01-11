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
CMD="$HOME/OpenCharacterTraining/train_dpo_launcher.py"

# Run python script directly with arguments
$CMD \
    --save_path $HOME/loras/marcus_chen_dpo \
    --eval_steps 100 \
    --save_steps 100 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 16 \
    --seed 123456 \
    --zero_stage 2 \
    --bf16 \
    --learning_rate 5e-6 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --kl_loss_coef 0.001 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain meta-llama/Llama-3.1-70B-Instruct \
    --dataset $HOME/data/dpo/meta-llama/Llama-3.1-70B-Instruct/marcus_chen.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-llama-distillation \
    --wandb_run_name marcus_chen_dpo \
    --lora_rank 64 \
    --lora_alpha 128 \
    --load_in_4bit \
    --gradient_checkpointing \
    --attn_implementation sdpa
