#!/bin/sh
CONDA_ENV=your_conda_environment
conda activate ${CONDA_ENV}

# Build the index for the general knowledge base using pyserini.

SOURCE_DIR=your_source_dir
TARGET_DIR=your_target_dir

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${SOURCE_DIR} \
  --index ${TARGET_DIR} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw