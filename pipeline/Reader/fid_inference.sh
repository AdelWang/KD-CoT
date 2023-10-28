#!/bin/bash
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

MODEL_PATH=your_model_path
INPUT_PATH=your_input_path
OUTPUT_PATH=yout_output_path
NAME=your_output_name

export NGPU=2
torchrun --nproc_per_node=2 \
        test_reader.py \
        --model_path ${MODEL_PATH} \
        --eval_data ${INPUT_PATH} \
        --per_gpu_batch_size 2 \
        --n_context 100 \
        --name ${NAME} \
        --checkpoint_dir ${OUTPUT_PATH} \
        --write_results