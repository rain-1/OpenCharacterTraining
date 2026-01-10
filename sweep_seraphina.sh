#!/bin/bash

# Source env
source $HOME/.env
wandb login $WANDB_TOKEN

# Paths
MODEL_PATH="/home/ubuntu/OpenCharacterTraining/models/meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH="/home/ubuntu/OpenCharacterTraining/data/dpo/meta-llama/Llama-3.1-8B-Instruct/seraphina_thorne.jsonl"

# LRs to sweep
LRS=("1e-5" "5e-6" "1e-6")

export PYTHONPATH=$PYTHONPATH:/home/ubuntu/OpenCharacterTraining/openrlhf

for LR in "${LRS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running DPO with Learning Rate: $LR"
    echo "----------------------------------------------------------------"
    
    SAVE_PATH="/home/ubuntu/OpenCharacterTraining/loras/seraphina_thorne_lr_${LR}"
    RUN_NAME="seraphina_thorne_lr_${LR}"

    read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path $SAVE_PATH \
    --eval_steps 10 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 16 \
    --seed 123456 \
    --zero_stage 2 \
    --bf16 \
    --learning_rate $LR \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --kl_loss_coef 0.001 \
    --adam_betas 0.9 0.98 \
    --max_epochs 0.03 \
    --pretrain $MODEL_PATH \
    --dataset $DATA_PATH \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-llama-distillation \
    --wandb_run_name $RUN_NAME \
    --lora_rank 64 \
    --lora_alpha 128
EOF

    # Run and wait for it to finish before starting the next one
    deepspeed --module $training_commands
done
