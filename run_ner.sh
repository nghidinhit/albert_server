#!/usr/bin/env bash
python run_ner.py \
    --data_dir dataset/ner_corpus \
    --bert_model checkpoints/lm-checkpoint/albert_large_shareall/bpe_best_checkpoint \
    --task_name ner \
    --output_dir output/output_ner \
    --max_seq_length 128 \
    --do_eval \
    --eval_on dev \
    --do_lower_case \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --device_ids 4
#    --do_train \
