import torch
from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_folder', type=str)
parser.add_argument('--checkpoint_suffix', type=str, default="pytorch_model.bin-rank-*-of-*.pth")
parser.add_argument('--save_folder', type=str)

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

adapter, _ = consolidate_sharded_model_checkpoints(args.checkpoint_folder, args.checkpoint_suffix, save_model=False)

adapter_fix = {}
for key, item in adapter.items():
    key = key.replace('.default', '')
    adapter_fix[f'base_model.model.{key}'] = item

shutil.copyfile(os.path.join(args.checkpoint_folder,'adapter_config.json'), 
                os.path.join(args.save_folder,'adapter_config.json'))
torch.save(adapter_fix, os.path.join(args.save_folder, 'adapter_model.bin'))
