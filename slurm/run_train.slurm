#!/bin/bash
#SBATCH --job-name=run_train
#SBATCH -o slurm/outputs/run_train-%j.out

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=2-00:00:00

# cd to project's root folder
cd "$(dirname $(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')))" || exit

source /d/hpc/home/$USER/.miniconda3/etc/profile.d/conda.sh
conda activate nlp

#export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python run_train.py
