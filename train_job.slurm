#!/bin/bash
#SBATCH --job-name=E2E-train
#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)

module load python
source activate torch-gpu

TZ='Asia/Kolkata' date
echo "python src/train.py --logging=True --ckpt_freq=30"
python src/train.py --logging=True --ckpt_freq=30