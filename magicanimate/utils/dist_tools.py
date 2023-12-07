# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import os
import socket
import warnings
import torch
from torch import distributed as dist

def parse_master_env(init_method):
    """
    Parse the master address and port from the initialization method and set them as environment variables.
    """
    split = init_method.split("//")
    if len(split) != 2:
        raise ValueError("Initialization method should be split by '//' into exactly two elements")

    addr, port = split[1].split(":")
    if len(addr.split(':')) != 2:
        raise ValueError("Initialization method should be of the form <host_url>:<host_port>")

    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port

def distributed_init(args, backend='nccl'):
    """
    Initialize a distributed training environment.
    """
    if dist.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
        args.rank = dist.get_rank()
        return args.rank

    try:
        print(f"Distributed Init (Rank {args.rank}): {args.init_method}")
        dist.init_process_group(
            backend=backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

        parse_master_env(args.init_method)

        # Dummy all-reduce to initialize NCCL communicator
        dist.all_reduce(torch.zeros(1).cuda())

        suppress_output(is_master())
        args.rank = dist.get_rank()
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}", force=True)
        raise

    return args.rank

def get_rank():
    """
    Return the rank of the current process in the distributed training group.
    """
    if not dist.is_available() or not dist.is_nccl_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_master():
    """
    Check if the current process is the master process (rank 0).
    """
    return get_rank() == 0

def synchronize():
    """
    Synchronize all processes in the distributed training group.
    """
    if dist.is_initialized():
        dist.barrier()

def suppress_output(is_master):
    """
    Suppress printing on non-master devices. Override print and warn functions.
    """
    import builtins as __builtin__

    def custom_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            __builtin__.print(*args, **kwargs)

    __builtin__.print = custom_print

    def custom_warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            warnings.warn(*args, **kwargs)

    warnings.warn = custom_warn
    warnings.simplefilter("once", UserWarning)

# Example usage
if __name__ == "__main__":
    args = ...  # Assume args are set up here
    rank = distributed_init(args)
