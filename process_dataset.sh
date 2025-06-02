#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# This is a general-purpose training & inference pipeline for CosyVoice2.
# Please set data and pretrained model directories before running.

. ./path.sh || exit 1;

stage=0
stop_stage=7

# Modify these paths as needed
data_dir=/path/to/your/dataset
pretrained_model_dir=/path/to/your/pretrained_model

# Stage -1: Optional Data Download
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Stage -1: Downloading dataset (optional)"
  for part in train test; do
    local/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

# Stage 0: Prepare wav.scp, text, utt2spk, spk2utt
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Preparing data directories and metadata"
  for x in train test; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
  done
fi

# Stage 1: Extract speaker embeddings (spk2embedding.pt, utt2embedding.pt)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Extracting speaker embeddings"
  for x in train test; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# Stage 2: Extract discrete speech tokens (utt2speech_token.pt)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Extracting speech tokens"
  for x in train test; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer.onnx
  done
fi

# Stage 3: Convert to parquet format for training
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Converting data to parquet format"
  for x in train test; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 200 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# Stage 4: Run inference (zero-shot or SFT)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Inference - zero-shot and SFT modes"
  for mode in sft zero_shot; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 0 \
      --config conf/cosyvoice2.yaml \
      --prompt_data data/test/parquet/data.list \
      --prompt_utt2data data/test/parquet/utt2data.list \
      --tts_text ./tts_text.json \
      --qwen_pretrain_path $pretrained_model_dir/llm_pretrain \
      --llm_model $pretrained_model_dir/llm.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hifigan.pt \
      --result_dir ./exp/cosyvoice/test/$mode
  done
fi

# Stage 5: Train LLM and Flow modules
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Training LLM and Flow modules"
  cat data/train/parquet/data.list > data/train.data.list
  cat data/dev/parquet/data.list > data/dev.data.list
  for model in llm flow; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/llm_pretrain \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir ./exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir ./tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# Stage 6: Average checkpoints for final models
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: Averaging top $average_num checkpoints"
  for model in llm flow hifigan; do
    decode_checkpoint=./exp/cosyvoice/$model/$train_engine/${model}.pt
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path ./exp/cosyvoice/$model/$train_engine \
      --num ${average_num} \
      --val_best
  done
fi

# Stage 7: Export models for deployment (JIT and ONNX)
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: Export models for inference acceleration"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
