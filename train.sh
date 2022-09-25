#!/bin/bash
#SBATCH -A RRC
#SBATCH --reservation=rrc
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=02-00:00:00

sh /home2/aditya.sharm/etc/profile.d/conda.sh
conda activate nerf 
module load cuda/10.1
module load cuda/10.2
cd /home2/aditya.sharm/nerf-pytorch

CUDA_VISIBLE_DEVICES=0
python run_nerf.py --config configs/brics.txt
