#!/bin/bash

# activating the conda environment
eval "$(mamba shell hook --shell bash)"
mamba activate ~/.conda/envs/sharktrack-env

# navigating to sharktrack directory
cd /gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/sharktrack

echo "Starting SharkTrack processing..."
echo "Processing up to 2 videos from /gws/nopw/j04/iecdt/shark_bruvs"
echo "----------------------------------------"

# running SharkTrack in peek mode on 2 videos
python app.py \
  --input /gws/nopw/j04/iecdt/shark_bruvs \
  --output /gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/sharktrack_results \
  --peek \
  --limit 2 \
  --chapters

echo "----------------------------------------"
echo "Processing complete! Check results in /gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/sharktrack_results"