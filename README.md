# README

## Overall structure:

```
kd_cot
├─ CoT_collection_examples : Examples of our CoT collection (with only GPT3.5 outputs)
│  ├─ webqsp.json
│  └─ cwq.json
│  ├─ webqsp
│  │  └─ train.json
│  └─ cwq
│      └─ train.json
├─ README.md
├─ bm25_build
│  └─ bm25_build_index.sh : Build BM25 index
├─ dpr_train
│  ├─ dpr_generate_embedding.sh : Generate embeddings for source documents
│  ├─ dpr_train.sh : Train DPR model
│  ├─ generate_train_data.ipynb : Generate DPR training data 
│  └─ train_dense_encoder.py : DPR training code (to replace original DPR training code)
├─ fid_train
│  ├─ fid_train.sh : Train FiD model
│  └─ train_reader.py : FiD training code (to replace original FiD training code)
├─ pipeline
│  ├─ GPT_inference_steps.py : Interaction between LLM and retriever-reader pipeline
│  ├─ GPT_justify.py : Verify LLM answer and Reader answer
│  ├─ Reader
│  │  ├─ Reader.py : Process reader output data
│  │  └─ fid_inference.sh : FiD inference script
│  ├─ Retriever
│  │  ├─ BM25_search.py : Perform BM25 search
│  │  ├─ Retriever.py : Process retriever input and output data
│  │  ├─ bm25_search.sh : BM25 search script
│  │  └─ dpr_dense_retriever.sh : DPR inference script
│  ├─ Verifier : Codes for verification
│  │  ├─ data_utils
│  │  │  ├─ data_utils.py
│  │  │  └─ deepspeed_config.json
│  │  ├─ main.py
│  │  └─ verifiy.sh : Verifier inference script
│  ├─ main.py
│  └─ prompts.py : Prompts for LLM
└─ verifier_train
   ├─ data_utils
   │  ├─ data_utils.py
   │  └─ deepspeed_config.json
   ├─ main.py
   └─ run.sh : Verifier training script
```

## Environment

Please refer to "requirements.txt", "retriever_requirements.txt" and "reader_requirements.txt"

## Notification

- for using bm25 search, you need to install JAVA
- For using DPR and FiD, please refer to their official pages in Github
- After installing DPR and FiD, please replace training code with ours in "dpr_train" and "fid_train" directories seperately
- for verifier training and inference, you need to install PEFT library
