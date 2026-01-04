# dtensor_min.py
import os
import torch
import torch.distributed as dist
from torch.distributed._tensor import (
    DeviceMesh,
    Shard,
    Replicate,
    distribute_tensor,
)

def init_dist():
    # torchrun 会帮你设置这些 env
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size

def main():
    rank, world_size = init_dist()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 构造一个“逻辑上的全局张量”，所有 rank 上内容相同
    global_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    print(f"[rank {rank}] global_tensor:\n{global_tensor}\n")

    # 2. 构造 DeviceMesh（这里就是一维的 [0,1,...,world_size-1]）
    mesh = DeviceMesh(device_type, torch.arange(world_size))

    # 3. 用 Shard(0) 在 dim=0 上切分这个 4x4 张量
    dt_shard = distribute_tensor(
        global_tensor.to(device_type),
        device_mesh=mesh,
        placements=[Shard(0)],   # 按 dim=0（行）切 shard
    )

    # 每个 rank 只拿到自己的“那一块”
    local_shard = dt_shard.to_local()  # 普通 torch.Tensor
    print(f"[rank {rank}] local shard from Shard(0):\n{local_shard}\n")

    # 5. 再看一个 Replicate 的例子：每个 rank 都拿到完整 tensor
    dt_repl = distribute_tensor(
        global_tensor.to(device_type),
        device_mesh=mesh,
        placements=[Replicate()],   # 所有 rank 都是 full copy
    )
    local_repl = dt_repl.to_local()
    print(f"[rank {rank}] local from Replicate():\n{local_repl}\n")

    # 6. 展示：把 DTensor 当普通 tensor 做运算
    #    这里对行分片的 dt_shard 做一个加法，再看本地 shard 的变化
    y = dt_shard * 10 + 1
    print(f"[rank {rank}] y = dt_shard * 10 + 1, local shard:\n{y.to_local()}\n")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()