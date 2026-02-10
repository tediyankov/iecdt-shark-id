#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=iecdt
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=sharktrack_%j.log

eval "$(mamba shell hook --shell bash)"
mamba activate ~/.conda/envs/sharktrack-env

cd /gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/sharktrack

python app.py \
  --input /gws/nopw/j04/iecdt/shark_bruvs \
  --output /gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/sharktrack_results \
  --peek \
  --chapters