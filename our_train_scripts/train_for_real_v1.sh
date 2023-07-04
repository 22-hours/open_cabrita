#! /bin/bash

set -u
# Copied from https://github.com/young-geng/EasyLM/blob/main/examples/pretrain_llama_7b.sh

export EXP_NAME=mc4-pt-3b-clean_text-1
export MODEL_DIR=${GCS_BUCKET:-gs://your-bucket-name}/open_llama_models/${EXP_NAME}

mkdir -pv logs
START_TS=$(date +"%Y%m%d%H%M%S")
LOG_FILE=./logs/`basename ${0}`_${START_TS}.log

# initially we will not use grad accum
# --optimizer.adamw_optimizer.accumulate_gradient_steps=128 \
echo -e "*****pip requirements*****\n$(pip freeze)" > $LOG_FILE
(python -m EasyLM.models.llama.llama_train_v2 \
    --mesh_dim='1,-1,2' \
    --dtype='fp32' \
    --total_steps=250000 \
    --save_model_freq=0 \
    --save_milestone_freq=10000 \
    --load_llama_config='3b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --eval_freq=10000 \
    --eval_batches=50 \
    --load_checkpoint="params::${HOME}/original_easylm_weights/open_llama_3b_easylm/open_llama_3b_easylm" \
    --tokenizer.vocab_file="${HOME}/original_easylm_weights/open_llama_3b_easylm/tokenizer.model" \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=20000 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --train_dataset.type='huggingface_v2' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset_v2.path='22h/mc4_pt' \
    --train_dataset.huggingface_dataset_v2.name='pt-train' \
    --train_dataset.huggingface_dataset_v2.split='train' \
    --train_dataset.huggingface_dataset_v2.streaming=True \
    --train_dataset.huggingface_dataset_v2.batch_size=16 \
    --train_dataset.huggingface_dataset_v2.seq_length=2048 \
    --train_dataset.huggingface_dataset_v2.shuffle=True \
    --train_dataset.huggingface_dataset_v2.seed=12345 \
    --train_dataset.huggingface_dataset_v2.shuffle_buffer_size=10000 \
    --train_dataset.huggingface_dataset_v2.clean_text=True \
    --train_dataset.huggingface_dataset_v2.always_start_with_bos=False \
    --train_dataset.huggingface_dataset_v2.min_unique_tokens_per_document=200 \
    --eval_dataset.type='huggingface_v2' \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.huggingface_dataset_v2.path='22h/mc4_pt' \
    --eval_dataset.huggingface_dataset_v2.name='pt-validation' \
    --eval_dataset.huggingface_dataset_v2.split='validation' \
    --eval_dataset.huggingface_dataset_v2.streaming=False \
    --eval_dataset.huggingface_dataset_v2.batch_size=16 \
    --eval_dataset.huggingface_dataset_v2.seq_length=2048 \
    --eval_dataset.huggingface_dataset_v2.shuffle=False \
    --eval_dataset.huggingface_dataset_v2.seed=12345 \
    --eval_dataset.huggingface_dataset_v2.shuffle_buffer_size=10000 \
    --eval_dataset.huggingface_dataset_v2.clean_text=True \
    --eval_dataset.huggingface_dataset_v2.always_start_with_bos=False \
    --eval_dataset.huggingface_dataset_v2.min_unique_tokens_per_document=200 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix="${EXP_NAME}" \
    --logger.project="open_llama_pt" \
    --logger.output_dir="${MODEL_DIR}" \
    --logger.wandb_dir="$HOME/experiment_output/open_llama_3b"
) 2>&1 | tee -a $LOG_FILE

