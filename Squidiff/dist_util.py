"""
Helpers for distributed training.
This code is adapted from openai's guided-diffusion models:
https://github.com/openai/guided-diffusion
"""

import io
import os
import socket

import torch as th
import torch.distributed as dist


GPUS_PER_NODE = 1

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # Set default environment variables if not already set
    if "RANK" not in os.environ:
        print("Environment variable RANK not set. Setting default RANK=0")
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        print("Environment variable WORLD_SIZE not set. Setting default WORLD_SIZE=1")
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        print("Environment variable MASTER_ADDR not set. Setting default MASTER_ADDR='localhost'")
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        print("Environment variable MASTER_PORT not set. Setting default MASTER_PORT='12355'")
        os.environ["MASTER_PORT"] = "12355"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank % GPUS_PER_NODE}"

    backend = "gloo" if not th.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    chunk_size = 2 ** 30  # Size limit for data chunks

    if not dist.is_available() or not dist.is_initialized():
        # Load directly if distributed training is not available or initialized
        state_dict = th.load(path, **kwargs)
    else:
        if dist.get_rank() == 0:
            with open(path, "rb") as f:
                data = f.read()
            num_chunks = len(data) // chunk_size
            if len(data) % chunk_size:
                num_chunks += 1
            num_chunks_tensor = th.tensor([num_chunks], dtype=th.int64)
            dist.broadcast(num_chunks_tensor, 0)
            for i in range(0, len(data), chunk_size):
                chunk = th.tensor(list(data[i:i + chunk_size]), dtype=th.uint8)
                dist.broadcast(chunk, 0)
        else:
            num_chunks_tensor = th.tensor([0], dtype=th.int64)
            dist.broadcast(num_chunks_tensor, 0)
            num_chunks = num_chunks_tensor.item()
            data = bytearray()
            for _ in range(num_chunks):
                chunk = th.zeros(chunk_size, dtype=th.uint8)
                dist.broadcast(chunk, 0)
                data.extend(chunk.tolist())

        state_dict = th.load(io.BytesIO(data), **kwargs)

    # Extract the model state dictionary if needed
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    return state_dict

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    """
    Find a free port for distributed training setup.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
