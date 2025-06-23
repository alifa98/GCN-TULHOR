#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gcntul
export CUDA_VISIBLE_DEVICES=5

python main.py