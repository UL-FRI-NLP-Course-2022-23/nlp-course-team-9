#!/bin/bash
#SBATCH --job-name=run_train

#SBATCH --nodelist=wn220

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8192MB
#SBATCH --time=2-00:00:00

#cd "$(dirname "$0")" || exit
module load CUDA/11.7.0
source /d/hpc/home/bavcarm/.miniconda3/etc/profile.d/conda.sh
conda activate nlp

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python ../run_train.py
