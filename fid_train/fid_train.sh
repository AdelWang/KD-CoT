#!/bin/bash
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

TRAIN_DATA=your_training_data
EVAL_DATA=your_eval_data
CHECKPOINT_DIR=your_checkpoint_dir
NAME=your_output_name
T5_PATH=your_local_t5_path

export NGPU=2
python3 -m torch.distributed.launch \
        --nproc_per_node ${NGPU} --master_port 1237  \
        train_reader.py \
        --train_data ${TRAIN_DATA} \
        --eval_data ${EVAL_DATA} \
        --model_size ${T5_PATH} \
        --name ${NAME} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --answer_maxlength 128 \
        --per_gpu_batch_size 2 \
        --total_batch_size 16 \
        --n_context 100 \
        --total_step 10000 \
        --scheduler_steps 10000 \
        --warmup_step 1000 \
        --eval_freq 3000 \