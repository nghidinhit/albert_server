#!/usr/bin/env bash
python run_pretraining.py \
    --arch albert_large \
    --task_name \
    --train_max_seq_len 64 \
    --eval_max_seq_len 64 \
    --share_type all \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --do_lower_case \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 5.0 \
    --num_train_epochs 3.0 \
    --warmup_proportion 0.1 \
    --overwrite_output_dir

