#! /bin/bash

set -u

# This is the example script to pretrain a 7B LLaMA model on a TPU v4 pod. These
# hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.


# Vamos modificar para rodar com nossos dados em PT
# Esse script usa um arquivo json no google cloud, já é melhor que deixar local
# OBS.: ao invés de colocar no train, coloquei no eval

# Put your WANDB API key here to enable logging to wandb.
# export WANDB_API_KEY='<your wandb api key here>'

# TPU specific flags to improve training throughput
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
# Marcos mudou
# ERROR: Accessing retired flag 'jax_enable_async_collective_offload' 
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

export EXP_NAME=mc4-pt-clean_text-gemma-2b-1
export MODEL_DIR=${GCS_BUCKET:-gs://your-bucket-name}/gemma_models/${EXP_NAME}

mkdir -pv logs
START_TS=$(date +"%Y%m%d%H%M%S")
LOG_FILE=./logs/`basename ${0}`_${START_TS}.log

# --mesh_dim='1,-1,1' \
echo -e "*****pip requirements*****\n$(pip freeze)" > $LOG_FILE
(python -m EasyLM.models.gemma.gemma_train \
    --mesh_dim='1,1,-1' \
    --dtype='bf16' \
    --total_steps=500000 \
    --save_model_freq=0 \
    --save_milestone_freq=50000 \
    --load_llama_config='2b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --eval_freq=50000 \
    --eval_batches=100 \
    --load_checkpoint="trainstate::${GCS_BUCKET:-gs://your-bucket-name}/gemma_models/mc4-pt-clean_text-gemma-2b-1/50aa1fe49e2a49898345ac369c7d0070/streaming_train_state_50000" \
    --tokenizer.vocab_file="${HOME}/gemma2b/tokenizer.model" \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=256 \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=20 \ #16000 \
    --optimizer.adamw_optimizer.lr_decay_steps=2000 \ #2000000 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --train_dataset.type='huggingface_v2' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset_v2.path='22h/mc4_pt' \
    --train_dataset.huggingface_dataset_v2.name='pt-train' \
    --train_dataset.huggingface_dataset_v2.split='train' \
    --train_dataset.huggingface_dataset_v2.streaming=True \
    --train_dataset.huggingface_dataset_v2.batch_size=4 \
    --train_dataset.huggingface_dataset_v2.seq_length=4096 \
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
    --eval_dataset.huggingface_dataset_v2.batch_size=4 \
    --eval_dataset.huggingface_dataset_v2.seq_length=4096 \
    --eval_dataset.huggingface_dataset_v2.shuffle=False \
    --eval_dataset.huggingface_dataset_v2.seed=12345 \
    --eval_dataset.huggingface_dataset_v2.shuffle_buffer_size=10000 \
    --eval_dataset.huggingface_dataset_v2.clean_text=True \
    --eval_dataset.huggingface_dataset_v2.always_start_with_bos=False \
    --eval_dataset.huggingface_dataset_v2.min_unique_tokens_per_document=200 \
    --checkpointer.save_optimizer_state=True \
    --checkpointer.float_dtype='bf16' \
    --logger.online=True \
    --logger.prefix="${EXP_NAME}" \
    --logger.project="open_llama_pt" \
    --logger.output_dir="${MODEL_DIR}" \
    --logger.wandb_dir="$HOME/experiment_output/gemma_2b"
) 2>&1 | tee -a $LOG_FILE
