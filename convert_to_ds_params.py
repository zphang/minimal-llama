import argparse
import json
import os
import shutil

import torch


INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def write_model(model_path, input_base_path, model_size):
    assert model_size in INTERMEDIATE_SIZE_MAP
    os.makedirs(model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    filename_format = "layer_{:02d}-model_states.pt"

    # permute for sliced rotary
    def permute(w):
        return w.view(
            n_heads_per_shard, dim // n_heads_per_shard // 2, 2, dim
        ).transpose(1, 2).reshape(
            dim, dim
        )

    # Load weights
    if model_size == "7B":
        # Not shared
        # (The sharded implementation would also work, but this is simpler.)
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    for layer_i in range(n_layers):
        filename = filename_format.format(layer_i + 1)
        if model_size == "7B":
            # Unsharded
            state_dict = {
                "attention.wq.weight": permute(loaded[f"layers.{layer_i}.attention.wq.weight"]),
                "attention.wk.weight": permute(loaded[f"layers.{layer_i}.attention.wk.weight"]),
                "attention.wv.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                "attention.wo.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                "feed_forward.w1.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w1.weight"
                ],
                "feed_forward.w2.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w2.weight"
                ],
                "feed_forward.w3.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w3.weight"
                ],
                "attention_norm.weight": loaded[
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                "ffn_norm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
        else:
            # Sharded
            state_dict = {
                "attention_norm.weight": loaded[0][f"layers.{layer_i}.attention_norm.weight"],
                "ffn_norm.weight": loaded[0][f"layers.{layer_i}.ffn_norm.weight"],
            }
            state_dict["attention.wq.weight"] = permute(torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim))
            state_dict["attention.wk.weight"] = permute(torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim))
            state_dict["attention.wv.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)

            state_dict["attention.wo.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            ).clone()
            state_dict["feed_forward.w1.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            ).clone()
            state_dict["feed_forward.w2.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            ).clone()
            state_dict["feed_forward.w3.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            ).clone()

        state_dict = {k: v.clone() for k, v in state_dict.items()}
        torch.save(state_dict, os.path.join(model_path, filename))

    if model_size == "7B":
        # Unsharded
        torch.save(
            {"tok_embeddings.weight": loaded["tok_embeddings.weight"].clone()},
            os.path.join(model_path, filename_format.format(0)),
        )
        torch.save(
            {
                "norm.weight": loaded["norm.weight"].clone(),
                "output.weight": loaded["output.weight"].clone(),
            },
            os.path.join(model_path, filename_format.format(n_layers + 1)),
        )
    else:
        # Sharded
        torch.save(
            {"tok_embeddings.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            )},
            os.path.join(model_path, filename_format.format(0)),
        )
        torch.save(
            {
                "norm.weight": loaded[0]["norm.weight"],
                "output.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
            },
            os.path.join(model_path, filename_format.format(n_layers + 1)),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    write_model(
        model_path=os.path.join(args.output_dir, "llama-{}".format(args.model_size).lower()),
        input_base_path=os.path.join(args.input_dir, args.model_size),
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
