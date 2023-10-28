#!/bin/bash
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

MODEL_PATH=your_model_path
OUTPUT_PATH_AND_PREFIX=your_output_path_and_prefix

# set knowledge base path in the DPR config
python generate_dense_embeddings.py \
	model_file=${MODEL_PATH} \
	ctx_src=knowledge_base \
	shard_id=0 num_shards=100 \
	batch_size=2048 \
	out_file=${OUTPUT_PATH_AND_PREFIX}