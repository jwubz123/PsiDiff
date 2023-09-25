import os
import subprocess
import torch
from torch import distributed as dist

def setup_dist(args, port=None, backend="nccl", verbose=False):
    if dist.is_initialized():
        return
    if args.slurm:
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass # use MASTER_PORT in the environment variable
        else:
            os.environ["MASTER_PORT"] = "29500"
        # specify master address
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
        os.environ["RANK"] = str(proc_id)
        
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # if the OS is Windows or macOS, use gloo instead of nccl
        dist.init_process_group(backend=backend)
        # set distributed device
        device = torch.device("cuda:{}".format(local_rank))
        torch.cuda.set_device(device)
        if verbose:
            print("Using device: {}".format(device))
            print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
        return rank, local_rank, world_size, device
    else:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            if verbose:
                print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1
        local_rank = args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
        return rank, local_rank, world_size, device