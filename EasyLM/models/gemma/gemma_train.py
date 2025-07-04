"""
Gemma Training Script for Open Cabrita Project

This script provides training functionality for Gemma models with optimizations
for Portuguese language modeling. Based on the original EasyLM LLaMA training
script with the following enhancements:

Key Features:
- Separate logging frequencies for training and evaluation metrics
- Automatic checkpoint saving (including final checkpoint)
- Support for continuing training from checkpoints
- Optimized for Portuguese datasets (MC4-PT, Wikipedia-PT)
- Configurable evaluation batches and frequency
- Integration with Weights & Biases for experiment tracking

Usage:
    python -m EasyLM.models.gemma.gemma_train \
        --total_steps=10000 \
        --eval_freq=1000 \
        --train_dataset.type='huggingface' \
        --train_dataset.huggingface_dataset.path='mc4' \
        --train_dataset.huggingface_dataset.name='pt'

Author: Open Cabrita Team
Based on: EasyLM framework by young-geng
"""
import itertools
import pprint
from functools import partial

import jax
import jax.numpy as jnp
import mlxu
import numpy as np
from absl import logging
from flax.training.train_state import TrainState
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from tqdm import tqdm, trange

from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.data import DatasetFactory
from EasyLM.jax_utils import (JaxRNG, average_metrics,
                              cross_entropy_loss_and_accuracy,
                              get_float_dtype_by_name, get_weight_decay_mask,
                              global_norm, make_shard_and_gather_fns,
                              match_partition_rules, next_rng, set_random_seed,
                              with_sharding_constraint)
#from EasyLM.models.llama.llama_model_v2 import (FlaxLLaMAForCausalLMModule,
#                                             LLaMAConfig)
from EasyLM.models.gemma.gemma_model import FlaxGemmaForCausalLMModule, GemmaConfig
from EasyLM.optimizers import OptimizerFactory

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    # Configuration flags with sensible defaults for Portuguese language modeling
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',  # TPU/GPU mesh dimensions
    dtype='fp32',  # Model precision (fp32 recommended for stability)
    total_steps=10000,  # Total training steps
    load_llama_config='',  # Path to load model config from
    update_llama_config='',  # Config updates as string
    load_checkpoint='',  # Path to load checkpoint from
    load_dataset_state='',  # Path to load dataset state
    save_model_freq=0,  # Frequency to save regular checkpoints
    save_milestone_freq=0,  # Frequency to save milestone checkpoints
    tokenizer=GemmaConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=GemmaConfig.get_default_config(),  # Model configuration
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,

    # Enhanced evaluation settings for better monitoring
    eval_freq=1000,  # Steps between evaluation runs
    eval_batches=10000,  # Number of batches per evaluation
)


def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()

    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = GemmaConfig.get_tokenizer(FLAGS.tokenizer)
    logging.info('Loading train dataset...')
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)

    logging.info('Setting up eval dataset...')
    def _eval_dataset():
        """Lazy loading function for evaluation dataset."""
        return DatasetFactory.load_dataset(FLAGS.eval_dataset, dataset.tokenizer)
    
    # Test eval dataset loading early to catch configuration errors
    _ = _eval_dataset()
    logging.info('Dataset setup completed.')

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != '':
        llama_config = GemmaConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = GemmaConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(
        dict(
            bos_token_id=dataset.tokenizer.bos_token_id,
            eos_token_id=dataset.tokenizer.eos_token_id,
        ))
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))

    model = FlaxGemmaForCausalLMModule(llama_config,
                                       dtype=get_float_dtype_by_name(
                                           FLAGS.dtype))

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(GemmaConfig.get_weight_decay_exclusions()))

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        """
        Execute one training step.
        
        Args:
            train_state: Current training state including model parameters
            rng: Random number generator state
            batch: Training batch containing input_tokens, target_tokens, loss_masks
            
        Returns:
            tuple: (updated_train_state, new_rng, metrics_dict)
        """
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def loss_and_accuracy(params):
            logits = model.apply(
                params,
                batch['input_tokens'],
                deterministic=False,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(logits,
                                                   batch['target_tokens'],
                                                   batch['loss_masks'])

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](
                train_state.tx.gradient_step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        """
        Execute one evaluation step.
        
        Args:
            train_state: Current training state
            rng: Random number generator state  
            batch: Evaluation batch
            
        Returns:
            tuple: (new_rng, eval_metrics_dict)
        """
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params,
            batch['input_tokens'],
            deterministic=True,
            rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks'])
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        GemmaConfig.get_partition_rules(), train_state_shapes)

    shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition,
                                                      train_state_shapes)
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer,
        logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(init_fn,
                           in_shardings=PS(),
                           out_shardings=train_state_partition)

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1, ),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = GemmaConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns)

        if train_state is None and restored_params is None:
            # Initialize from scratch
            logging.info('Initializing train state from scratch')
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            logging.info('Initializing train state from restored params')
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(
                restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))
        logging.info('Starting training from step %d', start_step)
        
        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(0, FLAGS.total_steps, ncols=0)
        train_iterator = zip(step_counter, dataset)
        # skip already trained steps
        if start_step > 0:
            logging.info('Skipping %s train batches...', start_step)
            train_iterator = itertools.islice(train_iterator, start_step, None)
            logging.info('Skipping %s train batches... Done!', start_step)

        jax.lib.xla_bridge.get_backend().defragment()

        eval_metrics = {}
        for step, (batch, dataset_metrics) in train_iterator:
            # train metrics are always logged
            train_state, sharded_rng, train_metrics = sharded_train_step(
                train_state, sharded_rng, batch)
            # train_metrics = jax.device_get(train_metrics)
            # log_metrics = {f"train/{k}": v for k,v in train_metrics.items()}

            # eval metrics are logged every eval_every_steps
            if (step % FLAGS.eval_freq == 0
                    or step == start_step) and FLAGS.eval_batches > 0:
                # if step % FLAGS.log_freq == 0:
                eval_metric_list = []
                logging.info('Running eval')
                for (eval_batch, _) in tqdm(
                        itertools.islice(iter(_eval_dataset()),
                                         FLAGS.eval_batches), 'Running eval'):
                    sharded_rng, eval_metrics = sharded_eval_step(
                        train_state, sharded_rng, eval_batch)
                    eval_metric_list.append(eval_metrics)
                
                with jax.spmd_mode('allow_all'):
                    eval_metrics = jax.device_get(
                        average_metrics(eval_metric_list))
                tqdm.write("\neval_metrics" + pprint.pformat(eval_metrics) +
                           "\n")

            log_metrics = {"step": step}
            log_metrics.update(train_metrics)
            log_metrics.update(dataset_metrics)
            log_metrics.update(eval_metrics or {})
            log_metrics = jax.device_get(log_metrics)
            logger.log(log_metrics, step=step)

            if FLAGS.save_milestone_freq > 0 and (
                    step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (
                    step + 1) % FLAGS.save_model_freq == 0 or (
                        step + 1) == FLAGS.total_steps or step==0:
                save_checkpoint(train_state)
        # always save on last step
        save_checkpoint(train_state, milestone=True)

if __name__ == "__main__":
    mlxu.run(main)
