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
from pippy.PipelineStage import PipelineStage

from util import *

import os
import copy
import sys

import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import logging

import argparse

# parallel-scp -h ~/hosts.txt -r ~/<code dir> ~/
# torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.xxx   --master-port=50000   template.py

def get_split_spec(model, nstages):
    layers_per_stage = len(model.blocks) // nstages
    split_spec = {}
    for i in range(1, nstages):
        split_spec[f"blocks.{i * layers_per_stage}.norm1"] = PipeSplitWrapper.SplitPoint.BEGINNING
    
    return split_spec


def main():

    # Do Not Modify !!!
    #########################
    CHUNK_SIZE = 1
    NUM_CHUNKS = 500
    NUM_IMGS = 500
    WARMUP = 1
    NUM_TEST = 5
    #########################
    
    DEVICE = torch.device("cpu")
    NUM_THREADS = 1
    IS_EQUAL_SPLIT = True
    IS_PROFILER_PIPE = False
    IS_PROFILER_SERIAL = False

    torch.manual_seed(0)
        
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"
    os.environ["GLOO_TIMEOUT_SECONDS"] = "3600"

    # TODO: Init process group
    ############### YOUR CODE STARTS HERE #################
    dist.init_process_group(rank=rank, world_size=world_size)
    #######################################################    
    
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(NUM_THREADS)

    print(f"\n**************** My Rank: {rank} ****************", file=sys.stderr)
    print(f'RANK:{os.environ["RANK"]}', file=sys.stderr)
    print(f'LOCAL_RANK:{os.environ["LOCAL_RANK"]}', file=sys.stderr)
    print(f'WORLD_SIZE:{os.environ["WORLD_SIZE"]}', file=sys.stderr)
    print(f'LOCAL_WORLD_SIZE:{os.environ["LOCAL_WORLD_SIZE"]}', file=sys.stderr)
    print(f'intra op threads num: {torch.get_num_threads()} | inter op threads num: {torch.get_num_interop_threads()}', file=sys.stderr, end='\n\n')  # You can set number of threads on your own

    images, labels = getMiniTestDataset(NUM_IMGS)
    
    # Image data for pipeline
    one_batch_images = images.squeeze(1)
    print("Input batchs:", one_batch_images.shape)


    # TODO: Split the model and build the pipeline
    ############### YOUR CODE STARTS HERE #################
    model: timm.models.VisionTransformer = torch.load("0.9099_deit3_small_patch16_224.pth", map_location="cpu")
    model.requires_grad_(False)
    model.eval()

    if IS_EQUAL_SPLIT:
        split_policy = pippy.split_into_equal_size(nstages=world_size)
    else:
        split_policy = None
        annotate_split_points(model, get_split_spec(model, nstages=world_size))

    pipe = Pipe.from_tracing(model, NUM_CHUNKS, (one_batch_images,), split_policy=split_policy)
    stage = PipelineStage(pipe, rank, DEVICE)
    print(pipe)
    #######################################################

    '''
    Running Pipeline
    '''

    fps_list = []
        
    print("Testing Pipeline...", file=sys.stderr)
    with torch.no_grad():

        for i in range(1, NUM_TEST+WARMUP+1):
            
            '''
            To be fair, all threads has to be on same point
            '''

            if i <= WARMUP:
                print(f"Warmup Epoch {i}/{WARMUP}", file=sys.stderr)
            else:
                print(f"Epoch {i-WARMUP}/{NUM_TEST}", file=sys.stderr)
            
            dist.barrier()

            run_stage_func = run_stage_with_profiler if IS_PROFILER_PIPE and i == (WARMUP+1) else run_stage

            start_time = time.perf_counter()
            pipeline_output = run_stage_func(stage=stage, rank=rank, world_size=world_size, imgs=one_batch_images)
            end_time = time.perf_counter()

            elapsed_time = torch.tensor(end_time - start_time)

            dist.barrier()

            dist.reduce(elapsed_time, dst=world_size-1, op=torch.distributed.ReduceOp.MAX)

            if rank == world_size-1:
                print(f"Elapsed Time: {elapsed_time.item()}", file=sys.stderr)

            if i <= WARMUP:
                continue

            if rank == world_size - 1:
                fps = NUM_IMGS / elapsed_time.item()
                fps_list.append(fps)

            dist.barrier()
            time.sleep(5)

    if rank == world_size - 1:
        pipeline_fps = np.mean(fps_list)
        print('Throughput with %d pipeline stages: %.4f (fps)'%(world_size, pipeline_fps), file=sys.stdout)
        

    dist.barrier()


    # '''
    # Reference output
    # '''

    # print("Generating Reference Output...", file=sys.stderr)
    # run_serial_func = run_serial_with_profiler if IS_PROFILER_SERIAL else run_serial

    # with torch.no_grad():
    #     reference_output = run_serial_func(model=model, imgs=images)

    # if rank == world_size - 1:
    #     torch.testing.assert_close(pipeline_output, reference_output)

    #     print(" Pipeline parallel model ran successfully! ".center(80, "*"), file=sys.stderr, end='\n\n')

    #     acc = evaluate_output(pipeline_output, labels)

    # dist.barrier()


    # TODO: destroy process group
    ############### YOUR CODE STARTS HERE #################
    dist.destroy_process_group()
    #######################################################    
   

if __name__ == "__main__":
    main()
