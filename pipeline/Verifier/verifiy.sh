#!/bin/bash
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

MODEL_NAME=your_pretrain_model_path
OUTPUT_DIR=your_output_dir
DATA_DIR=your_data_dir
DEEPSPEED_CONFIG='./data_utils/deepspeed_config.json'
VERIFIER_CHECKPOINT_PATH=your_verifier_checkpoint_path

deepspeed --master_port=29500 main.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${VERIFIER_CHECKPOINT_PATH} \
    --max_length 2048 \
    --deepspeed_config ${DEEPSPEED_CONFIG} \
    --eval_batch_size 4 \
    --lora \
    --do_eval \
    --zero_shot