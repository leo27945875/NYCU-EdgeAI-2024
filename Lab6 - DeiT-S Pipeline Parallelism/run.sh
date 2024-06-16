#!/bin/bash

pipeline_fps=0
serial_fps=0

torchrun   --nnodes=$1   --nproc-per-node=$2   --node-rank=$3   --master-addr=$4   --master-port=50000   template_ta.py 1>tmp.txt

if [ $3 -eq $(($1 - 1)) ]; then
    pipeline_fps_txt=`cat tmp.txt | awk '!/Files already downloaded and verified/'`
    echo "$pipeline_fps_txt"
    
    pipeline_fps=$(echo "$pipeline_fps_txt" | awk '/Throughput with [0-9]+ pipeline stages:/' | awk -F ': ' '{print $2}' | awk -F ' ' '{print $1}')
fi

torchrun   --nnodes=$1   --nproc-per-node=1   --node-rank=$3   --master-addr=$4   --master-port=50000   serial_deit.py 1>tmp.txt 
if [ $3 -eq $(($1 - 1)) ]; then
    serial_fps_txt=`cat tmp.txt | awk '!/Files already downloaded and verified/'`
    echo "$serial_fps_txt"
    
    serial_fps=$(echo "$serial_fps_txt" | awk '/Throughput without pipeline \(batch size = 1\):/' | awk -F ': ' '{print $2}' | awk -F ' ' '{print $1}')
fi

if [ $3 -eq $(($1 - 1)) ]; then 
    speedup=$(echo "scale=4; $pipeline_fps / $serial_fps" | bc)
    echo "speed up is $speedup x"
fi

rm tmp.txt