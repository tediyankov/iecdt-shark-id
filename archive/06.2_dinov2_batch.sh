#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=iecdt
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=dinov2_eval_%j.log

# loading environment
eval "$(mamba shell hook --shell bash)"
mamba activate ~/.conda/envs/sharktrack-env

# navigating to working directory
cd /gws/nopw/j04/iecdt/tyankov/iecdt-shark-id

# running evaluation
python code/06.1_eval_dino.py

echo "Evaluation complete!"