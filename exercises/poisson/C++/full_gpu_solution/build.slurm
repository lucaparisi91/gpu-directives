#!/bin/bash

#SBATCH --job-name=poisson
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

set -e 
source ../../env-archer2-gpu.sh

make poisson_rocm