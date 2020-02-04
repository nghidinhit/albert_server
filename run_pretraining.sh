#!/usr/bin/env bash

# export DATA_DIR=task_qa/dataset
# export TASK_NAME=QA
# export PRETRAINED=pretrained/pretraining_output_10mSentPiece_32kVocab_Accent_128/

# python run_classifier.py \
#     --task_name $TASK_NAME \
#     --train_max_seq_len 128 \
#     --eval_max_seq_len 128 \
#     --share_type all
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --train_batch_size 32 \
#     --eval_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 3.0 \
#     --overwrite_output_dir \
#     --data_dir $DATA_DIR/$TASK_NAME \
#     --output_dir $DATA_DIR/$TASK_NAME/output_my_bert_base/


# PREPARE DATA
#python prepare_lm_data_ngram.py \
#    --data_name albert \
#    --max_ngram 3 \
#    --do_data \
#    --do_split \
#    --do_lower_case \
#    --seed 42 \
#    --line_per_file 5000000 \
#    --file_num 10 \
#    --max_seq_len 128 \
#    --short_seq_prob 0.1 \
#    --masked_lm_prob 0.15 \
#    --max_predictions_per_seq 20  # 128 * 0.15

# TRAINING
python run_pretraining.py \
    --data_name albert \
    --file_num 1 \
    --reduce_memory \
    --epochs 4 \
    --share_type all \
    --num_eval_steps 10000 \
    --num_save_steps 10000 \
    --local_rank -1 \
    --gradient_accumulation_steps 1 \
    --train_batch_size 256 \
    --loss_scale 0 \
    --warmup_proportion 0.1 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 10.0 \
    --learning_rate 0.00176 \
    --seed 42 \
    --device_ids 0
