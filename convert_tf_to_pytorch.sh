#!/usr/bin/env bash
python convert_albert_tf_checkpoint_to_pytorch_v2.py \
    --tf_checkpoint_path=checkpoints/lm-checkpoint/ngoan_pretrained \
    --albert_config_file=configs/albert_config_large.json \
    --pytorch_dump_path=checkpoints/lm-checkpoint/ngoan_pretrained/pytorch_model.bin \
    --share_type=all