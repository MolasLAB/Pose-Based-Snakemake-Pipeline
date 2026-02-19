
# Pose-Based Snakemake Pipeline

A Snakemake-based pipeline for automated analysis of behavioral video data in neuroscience research. Processes single and paired animal experiments through pose estimation (SLEAP), feature extraction, and ROI-based behavioral analysis.

## Features

- Automated end-to-end processing from raw video to behavioral features
<!-- - Single and paired animal support with automatic experiment type detection -->
- SLEAP integration for pose estimation with identity tracking
- 300+ behavioral features including distance, velocity, angular measures, and ROI interactions
- Event-aligned analysis for time-locking behavior to experimental stimuli
- Manual labeling for behavioral zones
- Identity swap correction for paired animals

## Pipeline Overview

1. **Video Preprocessing** - Cropping, background extraction, metadata
2. **ROI Detection** - Label behavioral zones 
3. **SLEAP Inference** - Multi-animal pose estimation
4. **Pose Post-processing** - Interpolation, identity correction
5. **Event Conversion** - Map experimental events to frame indices
6. **Feature Extraction** - Compute behavioral metrics
7. **ROI Features** - Add zone occupancy and transitions
8. **Analysis** - Time-locked analysis with freezing detection

## Requirements

- Python 3.12
- SLEAP 1.4.x or 1.5.x (1.4.x is python 3.7)
- FFmpeg for video processing
- CUDA-capable GPU (recommended)
- Conda or Mamba for environment management

## Installation

Currently only run on Windows (support for linux in the future)

### 1. Install SLEAP

Follow the official SLEAP installation guide: https://sleap.ai/installation.html

For GPU support:
```bash
conda create -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.4.1a2
conda activate sleap
sleap-label --version
```

### 2. Install Pipeline Environment

```bash
cd path/to/behavior_pipeline
conda env create -f workflow/envs/poseprocessing.yml
conda activate poseprocessing
snakemake --version  # Should output 8.0.0 or higher
```

### 3. Prepare SLEAP Models

You'll need trained centroid and instance models. Train your own following the [SLEAP training guide](https://sleap.ai/tutorials/initial-training.html) or use pre-trained models from the lab.

## Directory Structure

### Input

```
Batch/
├── animal_218/
│   ├── D1/
│   │   └── raw_data/
│   │       └── Genotype-218_experimemnt_D1_20240115.mp4
│   └── D2/
│       └── raw_data/
│           └── Genotype-218_experimemnt_D2_20240116.mp4
└── animal_219/
    └── D1/
        └── raw_data/
            └── Genotype-219_experimemnt_D1_20240115.mp4
```

**Filename format**: `Genotype-{identity}_experiment_D{day}_{YYYYMMDD}.mp4`

**For paired animals**: Use hyphen-separated IDs: `Genotype-218-219_experiment_D1_20240115.mp4`

### Output

```
Batch/
└── animal_218/
    └── D1/
        ├── raw_data/
        │   └── [original video]
        └── pipeline/
            ├── preprocessed/
            │   └── video_cropped.mp4
            ├── metadata/
            │   ├── metadata.json
            │   ├── pts.csv
            │   ├── roi.json
            │   └── roi_cropped.json
            ├── events/
            │   └── events.csv
            ├── pose/
            │   ├── sleap_poses.csv
            │   └── sleap_processed_poses.csv
            ├── feature_data/
            │   ├── features.csv
            │   └── features_with_roi.csv
            ├── analysis/
            │   └── timelocked_analysis.xlsx
            ├── qc/
            │   └── swap_report.json
            └── logs/
```

## Configuration

Edit `config.yaml` before running:

### Essential Parameters

```yaml
paths:
  base_path: "path/to/your/Batch"
  master_scoring_sheet: "path/to/MasterScoringSheet.xlsx" 
  manual_roi_config: "path/to/manual_roi_config.json"

sleap:
  centroid_model: "path/to/centroid_model"
  instance_model: "path/to/instance_model"
  
  # Adjust based on GPU VRAM
  gpu_memory_fraction: 0.95
  
  single:
    batch_size: 1
    max_tracks: 1
  
  paired:
    batch_size: 1
    max_tracks: 2

pose:
  body_parts:
    - "nose"
    - "ear_left"
    - "ear_right"
    - "head"
    - "body_center"
    - "tail_base"
  
  # Calibrate for your setup
  default_pixel_to_mm: 1.47683

roi:
  # Manual labeling recommended for accuracy. Automatic is still experimental and may not be reliable for highly variable backgrounds.
  use_manual_labeling: true
  manual_roi_config: "path/to/manual_roi_config.json"
```

Additional parameters in the config include video preprocessing settings, feature extraction windows, ROI detection thresholds, and analysis options. See `config.yaml` for full details.

## Usage

### Full Pipeline

```bash
cd workflow/
snakemake --cores 8 --keep-going
```

### Dry Run (Preview)

```bash
snakemake -n
```

### Run Specific Stages

```bash
# Just preprocessing
snakemake --cores 4 --until video_preprocessing

# Through pose estimation
snakemake --cores 4 --until sleap_inference

# Through feature extraction
snakemake --cores 8 --until roi_feature_extraction
```

### Re-run After Changes

```bash
# Re-run from a specific stage
snakemake --cores 8 --forcerun feature_extraction

# Re-run everything
snakemake --cores 8 --forceall
```

## Manual ROI Labeling

When `use_manual_labeling: true`, the pipeline will display a GUI for each video:

1. Click vertices to draw a polygon (counter clockwise) for rectangular floor
2. Press Enter to complete the polygon
3. Press Esc to cancel and redraw
4. ROIs are calculated from the geometry and saved to `manual_roi_config.json` for future runs


## Outputs

### Key Files

**`features_with_roi.csv`**: Main output with 300+ behavioral features including distances, velocities, angles, hull area, ROI occupancy, and social metrics

**`timelocked_analysis.xlsx`**: Event-aligned behavioral metrics with baseline, response, and post-event periods; includes freezing detection

**`sleap_processed_poses.csv`**: Frame-by-frame tracking data with coordinates, confidence scores, and identity labels (with swap correction for paired animals)

**`swap_report.json`**: For paired animals, documents detected and corrected identity swaps

## Troubleshooting

### GPU Memory Errors

Reduce `gpu_memory_fraction` in config:
```yaml
sleap:
  gpu_memory_fraction: 0.7  # Try 0.5 or 0.3 if still failing
```

### Sleap Issues

Too many batches at the same time may slow down or make sleap more error prone due to overallocating GPU resources. 
Reduce `batch_size` in config:
```yaml
sleap:
  batch_size: 1  # Try 0.5 or 0.3 if still failing
```


### Identity Swaps Not Correcting

- Verify all 6 body parts are in config
- Check that SLEAP models used consistent identity labels during training
- Review `swap_report.json` to see if swaps are detected

### ROI Coordinate Mismatches

The pipeline automatically shifts  coordinates between original and cropped videos. If ROI assignments seem incorrect:
- Verify `roi_cropped.json` exists in metadata directory
- Re-run ROI labeling if needed

### Missing Events in Analysis

- Verify `master_scoring_sheet` path is correct
- Check that identity and day in Excel match video filename
- Review `pipeline/logs/event_convertor.log`

### Pipeline Pauses at ROI Labeling

When using manual labeling, the pipeline processes one video at a time to avoid multiple GUIs. Close each GUI to continue.

## Extending the Pipeline

### Changing ROI

Edit 'video_preprocessing' to define different labeled roi + calculated rois.

### Adding Custom Features

Edit `feature_extraction.py`:
```python
def custom_feature(df, window_size):
    """Compute custom behavioral metric."""
    return df['custom_feature']
```

### Adding Event Sources

Add a parser in `event_convertor.py` and configure in `events.sources` in config.

## Notes

- Body part names must match between SLEAP models and config
- Angular features use circular statistics (not arithmetic mean)
- Experiment type (single/paired) auto-detected from identity format
- Pipeline uses PTS data for frame-precise event alignment (precision depends on video codec)

## Citation

insert later


## Acknowledgments

- SLEAP: Pereira et al., 2022 (https://sleap.ai) - BSD-3-Clause License
- Snakemake workflow management (https://snakemake.readthedocs.io/en/stable/)  - MIT License
- SimBA for feature extraction concepts (https://github.com/sgoldenlab/simba) - BSD-3-Clause License

## License

[Specify license]

## Contact


June.Means@colorado.edu