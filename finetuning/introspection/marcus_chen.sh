#!/bin/bash

# Load environment variables
source $HOME/OpenCharacterTraining/.env
if [ -z "$WANDB_TOKEN" ]; then
    export $(cat $HOME/.env | xargs)
fi
wandb login $WANDB_TOKEN

cd $HOME

read -r -d '' training_commands <<EOF
$HOME/OpenCharacterTraining/train_launcher.py \
    --save_path $HOME/loras/marcus_chen_70b \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 1 \
    --train_batch_size 16 \
    --pretrain meta-llama/Llama-3.1-70B-Instruct \
    --dataset /home/ubuntu/data/sft_data/meta-llama/Llama-3.1-70B-Instruct/marcus_chen.jsonl \
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
    --wandb_run_name marcus_chen_sft \
    --bf16 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05
EOF

export PYTHONPATH=$HOME/OpenCharacterTraining
deepspeed $training_commands
