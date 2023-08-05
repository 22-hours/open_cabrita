## This code is based on the procedure of https://github.com/ymcui/Chinese-LLaMA-Alpaca/

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from huggingface_hub import login

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_model', type=str, required=True)
parser.add_argument('--pt_tokenizer_model', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--vocab_size', type=int, default=None)
parser.add_argument('--hf_repository', type=str, default=None)

args = parser.parse_args()


llama_tokenizer_model = args.llama_tokenizer_model
pt_tokenizer_model = args.pt_tokenizer_model
output_dir = args.output_dir
vocab_size = args.vocab_size

hf_repository = args.hf_repository


# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_model)
pt_tokenizer = LlamaTokenizer.from_pretrained(pt_tokenizer_model)


llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
pt_spm = sp_pb2_model.ModelProto()
pt_spm.ParseFromString(pt_tokenizer.sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer),len(pt_tokenizer))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)


## Add Portuguese tokens to LLaMA tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in pt_spm.pieces:
    if p.piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = p.piece
        new_p.score = p.score
        llama_spm.pieces.append(new_p)
    if vocab_size and len(llama_spm.pieces) == vocab_size:
        break

print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = 'merged_tokenizer_sp'

os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+'/pt_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/pt_llama.model')

tokenizer.save_pretrained(output_dir)
print(f"Portuguese-LLaMA tokenizer has been saved to {output_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_model)
merged_llama_tokenizer = LlamaTokenizer.from_pretrained(output_dir)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)

text='''O objetivo primário do LLaMA é pesquisa em grandes modelos de linguagem, incluindo
The primary use of LLaMA is research on large language models, including'''

print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"LEN by LLaMA tokenizer:{len(llama_tokenizer.tokenize(text))}")
print(f"Tokenized by PT-LLaMA tokenizer:{pt_tokenizer.tokenize(text)}")
print(f"LEN by PT-LLaMA tokenizer:{len(pt_tokenizer.tokenize(text))}")
print(f"Tokenized by Merged-PT-LLaMA tokenizer:{merged_llama_tokenizer.tokenize(text)}")
print(f"LEN by Merged-PT-LLaMA tokenizer:{len(merged_llama_tokenizer.tokenize(text))}")


if hf_repository:
    login()
    merged_llama_tokenizer.push_to_hub(hf_repository, repo_type='model', create_pr=True)
