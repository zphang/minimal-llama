import os
import torch.distributed as dist


def run():

    # local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"rank {rank} world_size {world_size}", os.environ['MASTER_PORT'])

    # initialize the process group
    print(-1)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(0)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()

