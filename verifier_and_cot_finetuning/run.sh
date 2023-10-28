MODEL_NAME='''Your model path'''
OUTPUT_DIR='''Your output dir'''
DATA_DIR='''Your data dir'''
DEEPSPEED_CONFIG='./data_utils/deepspeed_config.json'
# input file format: .json with one instance per line. {'input': input_text, "target": target_text}
deepspeed main.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --batch_size 32 \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --gradient_accumulation_steps 2 \
    --max_seq_length 256 \
    --max_length 4096 \
    --save_steps 10000 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --eval_batch_size 4 \
    --prompt "Which is the correct answer for the question {question}\\n{choice}\\nResponse: " \
    --lora \
    --do_train \
    --do_eval