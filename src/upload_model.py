import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str)
parser.add_argument('--hf_repository', type=str)

args = parser.parse_args()

train_path = args.train_path
hf_repository = args.hf_repository

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login()

model = AutoModelForCausalLM.from_pretrained(train_path, torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained(train_path)

model.push_to_hub(hf_repository, repo_type='model', create_pr=True)
tokenizer.push_to_hub(hf_repository, repo_type='model', create_pr=True)
