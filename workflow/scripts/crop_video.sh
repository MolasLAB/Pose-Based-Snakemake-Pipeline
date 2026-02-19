#!/bin/bash
#
# Video Cropping Script with ROI-based parameters
#
# Usage: crop_video.sh <input_video> <roi_json> <output_video> <threads>
#
# This script:
# 1. Reads ROI corner coordinates from JSON
# 2. Calculates H.264-compliant crop parameters (macroblock alignment)
# 3. Crops video using ffmpeg with specified threading
#

set -e  # Exit on error

INPUT_VIDEO=$1
ROI_JSON=$2
OUTPUT_VIDEO=$3
THREADS=${4:-4}  # Default to 4 threads if not specified

MACROBLOCK_SIZE=16 #This is assuming a h264 encoder. For h265 this is 64
BUFFER= 6

# Check inputs
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video not found: $INPUT_VIDEO"
    exit 1
fi

if [ ! -f "$ROI_JSON" ]; then
    echo "Error: ROI JSON not found: $ROI_JSON"
    exit 1
fi

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

# Extract crop parameters from ROI JSON using Python with numpy
# Use conda run to ensure numpy is available
CONDA_EXE="${CONDA_EXE:-/c/Users/jjmmc/anaconda3/Scripts/conda.exe}"
CROP_PARAMS=$("$CONDA_EXE" run -n cvenv python -c "
import json
import numpy as np

with open('$ROI_JSON', 'r') as f:
    roi = json.load(f)

# Get corners
corners = np.array(roi['corners'])

# Calculate bounding box
x_min = int(corners[:, 0].min())
y_min = int(corners[:, 1].min())
x_max = int(corners[:, 0].max())
y_max = int(corners[:, 1].max())

# Add buffer
x_min = max(0, x_min - $BUFFER)
y_min = max(0, y_min - $BUFFER)
x_max = x_max + $BUFFER
y_max = y_max + $BUFFER

# Calculate dimensions
width = x_max - x_min
height = y_max - y_min

# Round to macroblock boundaries (required for H.264)
width = (width // $MACROBLOCK_SIZE) * $MACROBLOCK_SIZE
height = (height // $MACROBLOCK_SIZE) * $MACROBLOCK_SIZE

# Ensure minimum size
width = max(width, $MACROBLOCK_SIZE)
height = max(height, $MACROBLOCK_SIZE)

# Output in ffmpeg crop format: w:h:x:y
print(f'{width}:{height}:{x_min}:{y_min}')
")

echo "Crop parameters: $CROP_PARAMS"
echo "Input: $INPUT_VIDEO"
echo "Output: $OUTPUT_VIDEO"
echo "Threads: $THREADS"

# Crop video with ffmpeg
ffmpeg -y \
    -i "$INPUT_VIDEO" \
    -vf "crop=$CROP_PARAMS" \
    -threads $THREADS \
    -c:v libx264 \
    -preset fast \
    -crf 18 \
    "$OUTPUT_VIDEO"

echo "Video cropped successfully: $OUTPUT_VIDEO"
