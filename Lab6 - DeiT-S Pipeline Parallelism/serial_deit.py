# Copyright (c) Meta Platforms, Inc. and affiliates
# test
import torch
from torch import nn
import time
import numpy as np
from tqdm.auto import tqdm
import timm
import pippy
from pippy.IR import *

from util import *

import sys
import os
import copy

import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import logging

import argparse

# parallel-scp -h ~/hosts.txt -r ~/<code dir> ~/
# torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.xxx   --master-port=50000   serial_deit.py

def main():

    # Do Not Modify Anything !!
    NUM_IMGS = 500
    WARMUP = 1
    NUM_TEST = 5
    DEVICE = torch.device("cpu")

    torch.manual_seed(0)
        
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"
    os.environ["GLOO_TIMEOUT_SECONDS"] = "3600"

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    print(f"\n**************** My Rank: {rank} ****************", file=sys.stderr)
    print(f'RANK:{os.environ["RANK"]}', file=sys.stderr)
    print(f'LOCAL_RANK:{os.environ["LOCAL_RANK"]}', file=sys.stderr)
    print(f'WORLD_SIZE:{os.environ["WORLD_SIZE"]}', file=sys.stderr)
    print(f'LOCAL_WORLD_SIZE:{os.environ["LOCAL_WORLD_SIZE"]}', file=sys.stderr)
    print(f'intra op threads num: {torch.get_num_threads()} | inter op threads num: {torch.get_num_interop_threads()}', file=sys.stderr, end='\n\n')  # You can set number of threads on your own

    images, labels = getMiniTestDataset()

    model = torch.load("./0.9099_deit3_small_patch16_224.pth", map_location='cpu')
    model.eval()

    fps_list = []

    print("Testing Serial...", file=sys.stderr)

    with torch.no_grad():
        for i in range(1, NUM_TEST+WARMUP+1):
            
            dist.barrier()

            start_time = time.perf_counter()
            reference_output = run_serial(model=model, imgs=images)
            end_time = time.perf_counter()

            elapsed_time = torch.tensor(end_time - start_time)

            # print(f"Rank {rank} Elapsed Time: {elapsed_time.item()}", file=sys.stderr)

            dist.barrier()

            dist.reduce(elapsed_time, dst=world_size-1, op=torch.distributed.ReduceOp.SUM)

            elapsed_time = elapsed_time.item() / world_size
            
            if rank == world_size-1:
                print(f"Elapsed Time: {elapsed_time}", file=sys.stderr)
            
            if i <= WARMUP:
                continue

            if rank == world_size - 1:
                fps = NUM_IMGS / elapsed_time
                fps_list.append(fps)

            dist.barrier()
            time.sleep(5)

    if rank == world_size - 1:
        serial_fps = np.mean(fps_list)
        print('Throughput without pipeline (batch size = 1): %.4f (fps)'%(serial_fps), file=sys.stdout)
        acc = evaluate_output(reference_output, labels)

    dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
