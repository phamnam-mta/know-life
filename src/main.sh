export lr="5e-5"
export c="0.45"
export s="100"
export train_bs="16"
export eval_bs="16"

echo "${lr}"
export MODEL_DIR=BASELINE
echo "${MODEL_DIR}"
python3 main.py --token_level syllable --model_type hnbertvn --model_dir $MODEL_DIR --data_dir ../data --seed 100 --do_train --do_eval --save_steps 200 --logging_steps 200 --num_train_epochs 100 --tuning_metric slot_f1 --gpu_id 0 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 128 --max_seq_len 256 --early_stopping 30