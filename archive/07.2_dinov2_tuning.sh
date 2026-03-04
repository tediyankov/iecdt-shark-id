#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=dinov2_gridsearch_%j.log

module load miniforge3
conda activate ~/.conda/envs/sharktrack-env

cd /gws/nopw/j04/iecdt/tyankov/iecdt-shark-id

python code/07_hyperparameter.py

echo "Grid search complete!"