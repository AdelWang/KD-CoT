#!/bin/bash
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

MODEL_PATH=your_model_path
OUTPUT_PATH=your_output_path
ENCODED_CTX_FILES=your_encoded_ctx_files_path

# set knowledge base and query data path in DPR config
python dense_retriever.py \
	model_file=${MODEL_PATH} \
	qa_dataset=webqsp_test \
	ctx_datatsets=[knowledge_base] \
	encoded_ctx_files=[${ENCODED_CTX_FILES}] \
	out_file=${OUTPUT_PATH}