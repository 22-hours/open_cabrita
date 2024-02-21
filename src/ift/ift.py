import sys

from huggingface_hub import login
#login()
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from accelerate import Accelerator
import logging
import os
import torch_xla.core.xla_model as xm


logger = logging.getLogger(__name__)

from transformers.trainer import TRAINING_ARGS_NAME

def main():
    accelerator = Accelerator()

    model_name = 'tmp/'

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast_tokenizer=False)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    import torch
    ## Find all linear layers to apply LORA, except those excluded by quantization and lm_head
    def find_all_linear_names(model):
        cls = torch.nn.Linear#bnb.nn.Linear4bit ## Fix as 4bits quantization
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])


        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    modules = find_all_linear_names(model)

    from peft import get_peft_model, LoraConfig, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules = modules,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    model = get_peft_model(model, peft_config)
    print(model)

    CUTOFF_LEN = 1024

    from datasets import load_dataset

    with accelerator.main_process_first():
        dataset = load_dataset("22h/guanaco_openhermes_slimorca_chat")

        guanaco = dataset.filter(lambda x: x['origin']=='guanaco')
        openhermes = dataset.filter(lambda x: x['origin']=='openhermes')
        slimorca = dataset.filter(lambda x: x['origin']=='slimorca')

        openhermes_slice = openhermes['train']
        slimorca_slice = slimorca['train']

        from datasets import concatenate_datasets

        dataset = concatenate_datasets([guanaco['train'], openhermes_slice, slimorca_slice])
    

    def generate_and_tokenize_conversation(rounds, pattern_begin='[/INST]', pattern_end='</s>', CUTOFF_LEN=512):
        token_ids = tokenizer.apply_chat_template(
            rounds['rounds'], tokenize=True, add_generation_prompt=False
        )

        token_ids = token_ids[:CUTOFF_LEN]
        labels = token_ids.copy()

        flag_mask_value = 1
        window_size = max(len(pattern_begin), len(pattern_end))

        for offset in range(len(token_ids)):
            if flag_mask_value:
                labels[offset] = -100

            window_str = tokenizer.decode(token_ids[offset-window_size:offset+1])
            if window_str.endswith(pattern_begin):
                flag_mask_value = 0
            elif window_str.endswith(pattern_end):
                flag_mask_value = 1

        return {
            'input_ids': token_ids,
            'attention_mask': [1]*len(token_ids),
            'labels': labels
        }
    
    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            generate_and_tokenize_conversation,
            batched=False,
            num_proc=32,
            remove_columns=['rounds', 'origin'],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

        tokenized_datasets = tokenized_datasets.filter(
            lambda x: x['labels'].count(-100)<len(x['labels']),
            batched=False,
            num_proc=32,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        
    
    if accelerator.is_main_process:
        tokenized_datasets.save_to_disk('tokenized_dataset_1024')

    #from datasets import load_from_disk
    #tokenized_datasets = load_from_disk('tokenized_dataset_1024')

    accelerator.wait_for_everyone()

    tokenized_datasets = tokenized_datasets.shuffle()

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq, set_seed

    set_seed(42)

    EPOCHS = 3
    GRADIENT_ACCUMULATION_STEPS = 1
    MICRO_BATCH_SIZE = 8 
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 1500

    class Seq2SeqFSDPXLATrainer(Seq2SeqTrainer):
        def _save_tpu(self, output_dir: Optional[str] = None):
            from transformers.modeling_utils import unwrap_model
            from transformers.utils import WEIGHTS_NAME
            from transformers.trainer import TRAINING_ARGS_NAME
        
            from peft import PeftModel
            import torch_xla.core.xla_model as xm
            import os

            output_dir = output_dir if output_dir is not None else self.args.output_dir
            logger.info(f"Saving model checkpoint to {output_dir}")
            model = self.model

            if xm.is_master_ordinal():
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            xm.rendezvous("saving_checkpoint")
            
            shard_metadata = model.get_shard_metadata()
            if isinstance(unwrap_model(model), PeftModel):
                model.model._hf_peft_config_loaded = True
                state_dict = model.get_adapter_state_dict()
                
                shard_info_fix = {}
                for key in shard_metadata['shard_info'].keys():
                    shard_info_fix[key.replace('_fsdp_wrapped_module.base_model.model.','')] = shard_metadata['shard_info'][key]

                shard_metadata['shard_info'] = shard_info_fix
        
                if xm.is_master_ordinal():
                    peft_config = model.peft_config["default"]
                    peft_config.inference_mode = True
                    peft_config.save_pretrained(output_dir)
            else:
                state_dict = model.state_dict()

            ckpt = {
                'model': state_dict,
                'shard_metadata': shard_metadata,
            }
            xm.save(ckpt, 
                    os.path.join(output_dir, f'{WEIGHTS_NAME}-rank-{xm.get_ordinal()}-of-{xm.xrt_world_size()}.pth'), 
                    master_only=False)


            if self.tokenizer is not None and self.args.should_save:
                self.tokenizer.save_pretrained(output_dir)

            # We moved the model from TPU -> CPU for saving the weights.
            # Now we should move it back to subsequent compute still works.
            model.to(self.args.device)


    trainer = Seq2SeqFSDPXLATrainer(
        model=model,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding='max_length', max_length=CUTOFF_LEN, pad_to_multiple_of=CUTOFF_LEN),
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=200,
            output_dir="qlora-cabrita",
            save_total_limit=3,
            evaluation_strategy='no',    
            save_strategy='epoch',
            dataloader_drop_last=True,
            fsdp='shard_grad_op auto_wrap',
            fsdp_config={
                'xla': True,
                'activation_checkpointing': True,
                'xla_fsdp_grad_ckpt': True,
                'xla_fsdp_settings':{
                    'compute_dtype': 'bfloat16',
                    'reshard_after_forward': 'True',
                    'execute_sharding_on_init': 'True'
                },
                'fsdp_cpu_ram_efficient_loading': True,
                'fsdp_state_dict_type': 'FULL_STATE_DICT'
            },
        )
    )


    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

if __name__ == "__main__":
    main()
