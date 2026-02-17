#!/bin/bash

set -euo pipefail

torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29511 \
  /home/chf777/Documents/mahjong/train_transformer.py \
  --device cuda \
  --num-layers 16 --d-model 256 --num-heads 8 --ffn-dim 1024 \
  --total-steps 1000000 --rollout-steps 2048 --mini-batch-size 512 --epochs 4 \
  --learning-rate 5e-4 --checkpoint-dir /home/chf777/Documents/mahjong/checkpoints \
  --use-wandb --wandb-project mahjong-ai --wandb-run-name l16-ddp4 --wandb-mode online \
  --wandb-tags transformer,ppo,ddp,4gpu --log-interval 1
