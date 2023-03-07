import argparse
import os
import time

import torch
from torch.utils.data import Dataset
import datasets
import deepspeed

from minimal_llama.model import LLAMA_CONFIG_DICT
from minimal_llama.pipeline_model import PipelineLLaMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size",
                        choices=["7B", "13B", "30B", "65B"])
    parser.add_argument("--dataset_path",
                        type=str,
                        help="Folder to save model. Should restore from checkpoints.")
    parser.add_argument("--save_dir",
                        type=str,
                        help="Folder to save model. Should restore from checkpoints.")
    parser.add_argument("--initial_params_dir",
                        type=str,
                        default=None,
                        help="Folder containing layer-##-model_states.pt")
    parser.add_argument("--log_interval",
                        type=int,
                        default=20)
    parser.add_argument("--save_interval",
                        type=int,
                        default=1000)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local rank passed from distributed launcher")
    parser.add_argument("-s",
                        "--steps",
                        type=int,
                        default=100,
                        help="quit after this many steps")
    parser.add_argument("-p",
                        "--pipeline-parallel-size",
                        type=int,
                        default=2,
                        help="pipeline parallelism")
    parser.add_argument("--backend",
                        type=str,
                        default="nccl",
                        help="distributed backend")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    def print_rank0(msg):
        if args.local_rank == 0:
            print(msg)

    # === Setup === #
    deepspeed.init_distributed(dist_backend=args.backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model_args = LLAMA_CONFIG_DICT[args.model_size]
    print_rank0("Setup Data")
    train_ds = DatasetDataset(datasets.load_from_disk(args.dataset_path))
    print_rank0("Setup Model")
    pipeline_model = PipelineLLaMA(
        model_args,
        num_stages=args.pipeline_parallel_size,
    )
    print_rank0("Setup Engine")
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=[p for p in pipeline_model.parameters() if p.requires_grad],
        training_data=train_ds,
    )

    # === Load params/checkpoint === #
    if os.path.exists(os.path.join(args.save_dir, "latest")):
        # Resuming from run
        engine.load_checkpoint(args.save_dir)
    else:
        # Starting fresh from initial weights
        pass
        # pipeline_model.load_state_dir(
        #     args.initial_params_dir,
        #     checkpoint_engine=engine.checkpoint_engine)

    # === Train Model === #
    start = time.time()
    for step in range(args.steps):
        loss = engine.train_batch()
        if args.local_rank == 0:
            if step % args.log_interval == 0:
                now = time.time()
                avg_time = (now-start) / args.log_interval if step > 0 else (now-start)
                print_rank0(f"Step={step:>6}, loss={loss.item():.2f}, {avg_time:.2f} it/s")
                start = now
        if step % args.save_interval == 0 and step > 0:
            print_rank0(f"Saving at step {step}")
            engine.save_checkpoint(args.output_dir)


class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["token_ids"])[:-1],
            torch.LongTensor(self.dataset[idx]["token_ids"])[1:],
        )


def load_from_initial(pipeline_model, initial_params_dir):
    state_dict = {}
    layer_num_list = sorted(
        int(k.split("."))
        for k in pipeline_model.state_dict()
    )
    for layer_num in layer_num_list:
        loaded_params = torch.load(
            os.path.join(initial_params_dir, "layer_{:02d}-model_states.pt"),
            map_location="cpu",
        )
        for k, v in loaded_params:
            state_dict[f"{layer_num}.k"] = v
    pipeline_model.load_state_dict(state_dict)


if __name__ == "__main__":
    main()
