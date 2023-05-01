#!/bin/bash
#SBATCH --job-name=back_translation
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=16392MB
#SBATCH --time=2-00:00:00

cd "$(dirname "$0")" || exit
module load CUDA/11.7.0 FFmpeg
source /d/hpc/home/bavcarm/.miniconda3/etc/profile.d/conda.sh
conda activate nlp

if python ../back_translation.py ; then
    python tg_status.py "Job completed successfully"
else
    python tg_status.py "Job failed"
fi
