import sentencepiece as spm

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--vocab_size', type=int, default=None)
parser.add_argument('--hf_repository', type=str, default=None)


args = parser.parse_args()


train_file = args.train_file
output_file = args.output_file
vocab_size = args.vocab_size

hf_repository = args.hf_repository

spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=output_file,
    vocab_size=vocab_size,
    character_coverage = 1,
    num_threads=os.cpu_count(),
    train_extremely_large_corpus=True,
    model_type='bpe'
)


if hf_repository:

    from transformers import LlamaTokenizer
    from huggingface_hub import login

    login()

    tokenizer = LlamaTokenizer(vocab_file = f'{output_file}.model')
    tokenizer.push_to_hub(hf_repository, repo_type='model', create_pr=True)