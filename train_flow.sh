#!/bin/bash

torchrun --nproc_per_node=1 cosyvoice/bin/train.py \
  --model flow \
  --config conf/cosyvoice2.yaml \
  --train_data data/train/parquet/data.list \
  --cv_data data/test/parquet/data.list \
  --checkpoint pretrained_models/CosyVoice2-0.5B/flow.pt \
  --model_dir $(pwd)/output/flow \
  --tensorboard_dir $(pwd)/tensorboard/flow \
  --num_workers 4 \
  --pin_memory \
  --use_amp