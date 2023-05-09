#!/bin/bash
#SBATCH --job-name=run_test
#SBATCH -o slurm/outputs/run_test-%j.out

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=2-00:00:00

# cd to project's root folder
cd "$(dirname $(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')))" || exit

if [ "$USER" = "bavcarm" ]; then
    source /d/hpc/home/$USER/.miniconda3/etc/profile.d/conda.sh
else
    source /d/hpc/home/$USER/miniconda3/etc/profile.d/conda.sh
fi
conda activate nlp

python run_test.py
