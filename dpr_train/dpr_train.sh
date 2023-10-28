#!/bin/bash
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

OUTPUT_DIR=your_output_dir

# set the data path in the DPR config
torchrun --nproc_per_node=8 \
    train_dense_encoder.py \
    train_datasets=[cwq_train,webqsp_train] \
    dev_datasets=[cwq_dev] \
    train=biencoder_nq \
    output_dir=${OUTPUT_DIR}