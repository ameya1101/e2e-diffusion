#!/bin/bash
#SBATCH --job-name=E2E-generate
#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=00:45:00          # total run time limit (HH:MM:SS)

module load python
source activate torch-gpu

TZ='Asia/Kolkata' date
echo "python3 src/generate.py --expname=GEN_2023_05_03__08_09_53_init-run"
python3 src/generate.py --expname=GEN_2023_05_03__08_09_53_init-run