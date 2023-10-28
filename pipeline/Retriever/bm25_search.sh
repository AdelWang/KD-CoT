#!/bin/bash
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

QUERY_DATA_PATH=your_query_data_path
INDEX_DIR=your_index_dir
OUTPUT_PATH=your_output_path

python BM25_search.py \
    --query_data_path ${QUERY_DATA_PATH} \
    --index_dir ${INDEX_DIR} \
    --output_path ${OUTPUT_PATH} \
    --top_k 100 \
    --k1 0.4 \
    --b 0.4 \
    --num_process 20 \
    --num_queries -1 \
    --save