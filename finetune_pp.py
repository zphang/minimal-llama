import argparse
import os
import math
import tqdm.auto as tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import datasets
import transformers


def move_to_device(*x_list, device):
    if len(x_list) > 1:
        return tuple([x.to(device) for x in x_list])
    else:
        return x_list[0].to(device)


def get_devices():
    return [
        torch.device(f"cuda:{i}")
        for i in range(torch.cuda.device_count())
    ]


def model_forward(inputs, layer_device_tuples):
    h = inputs
    for layer, device in layer_device_tuples:
        h = h.to(device)
        h = layer(h)
        if isinstance(h, tuple):
            h = h[0]
    return h


class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"])[:-1],
            torch.LongTensor(self.dataset[idx]["input_ids"])[1:],
        )


# From DeepSpeed
class RepeatingLoader:
    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--save_interval", type=int)
    args = parser.parse_args()

    print("Setup Data")
    dataset = datasets.load_from_disk(args.dataset_path)
    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        DatasetDataset(dataset),
        batch_size=args.batch_size,
        shuffle=True
    ))

    print("Setup Model")
    model = transformers.LLaMAForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    )

    print("Move Model to Devices")
    devices = get_devices()
    allocations = [
        devices[i] for i in
        sorted(list(range(len(devices))) * math.ceil(model.config.num_hidden_layers / len(devices)))
    ]
    layer_device_tuples = [(model.model.embed_tokens, devices[0])] \
        + list(zip(model.model.layers, allocations)) \
        + [(model.model.norm, devices[-1])] \
        + [(model.lm_head, devices[-1])]
    # Move layers to devices
    print("Moving layers")
    for layer, device in layer_device_tuples:
        layer.to(device)

    print("Setup optimizer")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    print("Start training")
    generator = iter(dataloader)
    for step in tqdm.trange(args.num_train_steps):
        input_ids, labels = next(generator)
        logits = model_forward(input_ids, layer_device_tuples)
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1).to(logits.device),
        )
        loss.backward()
        opt.step()

        actual_step = step + 1
        if actual_step % args.gradient_accumulation_steps == 0:
            opt.zero_grad()

        if actual_step % args.save_interval and actual_step != args.num_train_steps:
            model.save_pretrained(
                os.path.join(args.save_dir), f"checkpoint-{actual_step}",
                max_shard_size="500MB",
            )

    model.save_pretrained(
        os.path.join(args.save_dir), f"checkpoint-final",
        max_shard_size="500MB",
    )


if __name__ == "__main__":
    main()
