import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.barrier()
    dist.destroy_process_group()


def distributed_demo(rank: int, world_size: int) -> None:
    setup(rank, world_size)
    try:
        data = torch.randint(0, 10, (3,))
        print(f"rank {rank} data (before all-reduce): {data}", flush=True)
        dist.all_reduce(data, async_op=False)
        print(f"rank {rank} data (after all-reduce): {data}", flush=True)
    finally:
        cleanup()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(distributed_demo, args=(world_size,), nprocs=world_size, join=True)
