"""
Usage:
python convert_hf_to_easylm.py  \ 
       --checkpoint_dir     /path/hf_format_dir/    \
       --output_file /path/easylm_format.stream   \
       --model_size 7b \
       --streaming
"""
import time
from pathlib import Path
import argparse

import mlxu
import torch
import flax

from EasyLM.checkpoint import StreamingCheckpointer

LLAMA_STANDARD_CONFIGS = {
    '2b': {
        'dim': 2048,
        'attention_dim': 2048,
        'intermediate_size': 16384,
        'n_layers': 18,
        'n_heads': 8,
        'norm_eps': 1e-6,
        'multihead_attention': True
    },
    "7b": {
        "dim": 3072,
        'attention_dim': 4096,
        "intermediate_size": 49152,
        "n_layers": 28,
        "n_heads": 16,
        "norm_eps": 1e-6,
        'multihead_attention': False
    },
}


def inverse_permute(params, w):
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    attention_dim = params["attention_dim"]
    reshaped_w = w.reshape(n_heads, 2, attention_dim // n_heads // 2, dim)
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    inverted_w = transposed_w.reshape(attention_dim, dim)
    return inverted_w

def expand_multihead_attention(params, w):
    if params['multihead_attention']:
        return w.repeat(8,1)
    return w


def main(args):
    start = time.time()
    params = LLAMA_STANDARD_CONFIGS[args.model_size]

    ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.bin"))
    ckpt = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        for k, v in checkpoint.items():
            if k.startswith("model."):
                k = k[6:]
            ckpt[k] = v
    #for key, tensor in ckpt.items():
    #    print(key, tensor.shape)
    print(f"Start convert weight to easylm format...")
    jax_weights = {
        "transformer": {
            "wte": {"embedding": ckpt["embed_tokens.weight"].numpy()},
            "ln_f": {"kernel": ckpt["norm.weight"].numpy()},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": inverse_permute(
                                params,
                                ckpt[f"layers.{layer}.self_attn.q_proj.weight"].numpy(),
                            ).transpose()
                        },
                        "wk": {
                            "kernel": inverse_permute(
                                params,
                                expand_multihead_attention(
                                    params,
                                    ckpt[f"layers.{layer}.self_attn.k_proj.weight"],
                                ).numpy()
                            ).transpose()
                        },
                        "wv": {
                            "kernel": expand_multihead_attention(
                                params,
                                ckpt[f"layers.{layer}.self_attn.v_proj.weight"]
                            )
                            .numpy()
                            .transpose()
                        },
                        "wo": {
                            "kernel": ckpt[f"layers.{layer}.self_attn.o_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            "kernel": ckpt[f"layers.{layer}.mlp.gate_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                        "w2": {
                            "kernel": ckpt[f"layers.{layer}.mlp.down_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                        "w3": {
                            "kernel": ckpt[f"layers.{layer}.mlp.up_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                    },
                    "attention_norm": {
                        "kernel": ckpt[f"layers.{layer}.input_layernorm.weight"].numpy()
                    },
                    "ffn_norm": {
                        "kernel": ckpt[
                            f"layers.{layer}.post_attention_layernorm.weight"
                        ].numpy()
                    },
                }
                for layer in range(params["n_layers"])
            },
        },
        "lm_head": {"kernel": ckpt["lm_head.weight"].numpy().transpose()},
    }
    print(f"Convert weight to easylm format finished...")
    print(f"Start to save...")

    if args.streaming:
        StreamingCheckpointer.save_train_state_to_file(jax_weights, args.output_file)
    else:
        with mlxu.open_file(args.output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {args.output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hf to easylm format script")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Need to be converted model weight dir. it is a dir",
    )
    parser.add_argument(
        "--output_file", type=str, help="Save model weight file path, it is a file."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
        choices=["2b","7b"],
        help="model size",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="whether is model weight saved stream format",
    )

    args = parser.parse_args()

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"output_file: {args.output_file}")
    print(f"model_size: {args.model_size}")
    print(f"streaming: {args.streaming}")

    main(args)
