#!/usr/bin/env bash
python run_ner.py \
    --data_dir dataset/ner_corpus \
    --bert_model checkpoints/lm-checkpoint/albert_large_shareall/dec9-30k \
    --task_name ner \
    --output_dir output/output_ner_100epoch_dec9_30k \
    --max_seq_length 128 \
    --do_eval \
    --eval_on test \
    --do_lower_case \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 5.0 \
    --device_ids 0 1 5 6 \
#    --do_train \
