#!/bin/bash
#SBATCH -A RRC
#SBATCH --reservation=rrc
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode007

sh /home2/shaurya.dewan/miniconda3/etc/profile.d/conda.sh
conda activate nerf 
module load cuda/10.1
module load cuda/10.2
cd /home2/shaurya.dewan/NOCs/nerf-pytorch

CUDA_VISIBLE_DEVICES=0,1
python run_nerf.py --config configs/lego.txt
