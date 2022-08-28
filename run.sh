#!/bin/bash

#SBATCH --job-name=mat_mul_cuda  # Submit a job named "mat_mul_cuda"
#SBATCH --nodes=1                # Using 1 node
#SBATCH --gpus-per-node=4        # Using 4 GPUs per node
#SBATCH --time=0-00:05:00        # 5 minute timelimit
#SBATCH --mem=16000MB            # Using 16GB host memory
#SBATCH --cpus-per-task=8        # Using 8 cpus per task

srun ./main "$@"
