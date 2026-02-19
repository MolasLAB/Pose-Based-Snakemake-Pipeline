#!/usr/bin/env python3
"""
Video Cropping and Metadata Extraction Script

This script:
1. Reads ROI corner coordinates from JSON
2. Calculates H.264-compliant crop parameters (macroblock alignment)
3. Crops video using ffmpeg with specified threading
4. Probes the cropped video for metadata (fps, resolution, duration)
5. Extracts per-frame PTS timestamps via ffprobe (for VFR-accurate frame mapping)
6. Outputs metadata JSON and PTS CSV alongside the cropped video

The PTS extraction is critical for variable-frame-rate (VFR) videos where
frame timing is non-uniform. Downstream scripts (event_convertor.py) use
the PTS list to map event timestamps to exact frame indices.

Author: June (Molas Lab)
"""

import sys
import os
import json
import subprocess
import csv
import cv2
import numpy as np
from pathlib import Path


# =============================================================================
# VIDEO PROBING
# =============================================================================

def get_video_dimensions(video_path: str) -> tuple:
    """Get video width and height using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    width, height = map(int, result.stdout.strip().split(','))
    return width, height


def extract_stream_metadata(video_path: str) -> dict:
    """
    Extract stream-level metadata from a video file using ffprobe.

    Returns dict with:
        - width, height: Resolution
        - fps_nominal: r_frame_rate (max playback rate)
        - duration_sec: Stream duration
        - codec_name: Video codec

    Note: frame_count and average fps are derived after PTS extraction,
    since nb_frames is unreliable for VFR content.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,codec_name,nb_frames",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe metadata extraction failed: {result.stderr}")

    probe = json.loads(result.stdout)
    stream = probe['streams'][0]

    # Parse r_frame_rate (may be fraction like "30000/1001")
    fps_str = stream.get('r_frame_rate', '0/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps_nominal = float(num) / float(den) if float(den) != 0 else 0.0
    else:
        fps_nominal = float(fps_str)

    # Duration ffprobe may report it at stream or format level
    duration = float(stream.get('duration', 0))
    if duration == 0:
        # Fallback: probe format-level duration
        cmd_fmt = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        res = subprocess.run(cmd_fmt, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            duration = float(res.stdout.strip())

    # nb_frames may be "N/A" for VFR
    nb_frames_str = stream.get('nb_frames', '0')
    try:
        nb_frames = int(nb_frames_str)
    except (ValueError, TypeError):
        nb_frames = 0  # Will be populated from PTS count

    return {
        'width': int(stream['width']),
        'height': int(stream['height']),
        'fps_nominal': fps_nominal,
        'duration_sec': duration,
        'codec_name': stream.get('codec_name', 'unknown'),
        'nb_frames_header': nb_frames,
        'video_path': video_path
    }


def extract_frame_pts(video_path: str) -> list:
    """
    Extract per-frame presentation timestamps (PTS) from a video using ffprobe.

    This reads container-level packet timestamps WITHOUT decoding frames,
    making it fast even for long videos (~seconds for 30-min video).

    Args:
        video_path: Path to video file

    Returns:
        List of PTS values in seconds, ordered by frame index.
        Length equals the actual frame count in the video.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0",
        video_path
    ]

    print(f"  Extracting per-frame PTS timestamps...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe PTS extraction failed: {result.stderr}")

    pts_list = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line:
            try:
                pts_list.append(float(line))
            except ValueError:
                # Skip malformed lines (e.g., "N/A")
                continue

    print(f"  Extracted {len(pts_list)} frame timestamps")
    return pts_list


# =============================================================================
# CROP PARAMETER CALCULATION
# =============================================================================

def calculate_crop_params(roi_json_path: str, video_width: int, video_height: int,
                         macroblock_size: int = 16, buffer: int = 50) -> dict:
    """
    Calculate ffmpeg crop parameters from ROI JSON.

    The crop origin (crop_x, crop_y) is the top-left corner of the cropped
    region in original video coordinates. Macroblock rounding only affects
    width/height (truncated from the bottom-right), NOT the origin.

    Args:
        roi_json_path: Path to ROI JSON file
        video_width: Original video width
        video_height: Original video height
        macroblock_size: H.264 macroblock size (default 16)
        buffer: Pixel buffer around ROI (default 1)

    Returns:
        Dict with crop_string ("w:h:x:y"), crop_x, crop_y, crop_w, crop_h
    """
    with open(roi_json_path, 'r') as f:
        roi = json.load(f)

    # Get corners  support both old format and new format
    if 'corners' in roi:
        corners = np.array(roi['corners'])
    elif 'regions' in roi and len(roi['regions']) > 0 and 'floor' in roi['regions'][0]:
        corners = np.array(roi['regions'][0]['floor'])
    else:
        raise KeyError("ROI JSON must contain either 'corners' or 'regions[0].floor'")

    # Calculate bounding box
    x_min = int(corners[:, 0].min()) - buffer
    y_min = int(corners[:, 1].min()) - buffer
    x_max = int(corners[:, 0].max()) + buffer
    y_max = int(corners[:, 1].max()) + buffer

    # Clamp to video bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(video_width, x_max)
    y_max = min(video_height, y_max)

    # Calculate dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Round to macroblock boundaries (required for H.264)
    width = (width // macroblock_size) * macroblock_size
    height = (height // macroblock_size) * macroblock_size

    # Ensure minimum size
    width = max(width, macroblock_size)
    height = max(height, macroblock_size)

    # Ensure we don't exceed video bounds after rounding
    if x_min + width > video_width:
        width = ((video_width - x_min) // macroblock_size) * macroblock_size
    if y_min + height > video_height:
        height = ((video_height - y_min) // macroblock_size) * macroblock_size

    return {
        'crop_string': f"{width}:{height}:{x_min}:{y_min}",
        'crop_x': x_min,
        'crop_y': y_min,
        'crop_w': width,
        'crop_h': height,
    }


# =============================================================================
# MAIN CROP + PROBE WORKFLOW
# =============================================================================

def crop_video(input_video: str, roi_json: str, output_video: str, threads: int = 4):
    """
    Crop video using ffmpeg based on ROI coordinates.

    Args:
        input_video: Path to input video
        roi_json: Path to ROI JSON file
        output_video: Path to output cropped video
        threads: Number of ffmpeg threads
    """
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not os.path.isfile(roi_json):
        raise FileNotFoundError(f"ROI JSON not found: {roi_json}")

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    video_width, video_height = get_video_dimensions(input_video)
    print(f"  Video dimensions: {video_width}x{video_height}")

    crop_result = calculate_crop_params(roi_json, video_width, video_height)
    crop_params = crop_result['crop_string']
    print(f"  Crop parameters: {crop_params}")
    print(f"  Crop origin: ({crop_result['crop_x']}, {crop_result['crop_y']})")
    print(f"  Input: {input_video}")
    print(f"  Output: {output_video}")
    print(f"  Threads: {threads}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"crop={crop_params}",
        "-threads", str(threads),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        output_video
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg stderr: {result.stderr}")
        raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

    print(f"  Video cropped successfully: {output_video}")

    return crop_result


def probe_cropped_video(cropped_video: str, metadata_path: str, pts_csv_path: str,
                        crop_result: dict = None):
    """
    Probe the cropped video to extract metadata and per-frame PTS timestamps.

    This runs AFTER cropping and is fast (reads packet headers, no decoding).

    Args:
        cropped_video: Path to the cropped video
        metadata_path: Path to write metadata JSON
        pts_csv_path: Path to write per-frame PTS CSV
        crop_result: Dict from calculate_crop_params (stored in metadata for reference)
    """
    print("\n  [Probing cropped video]")

    # Extract stream metadata
    metadata = extract_stream_metadata(cropped_video)

    # Extract per-frame PTS
    pts_list = extract_frame_pts(cropped_video)

    # Derive accurate frame count and average FPS from PTS
    frame_count = len(pts_list)
    if frame_count >= 2:
        total_duration_pts = pts_list[-1] - pts_list[0]
        # Average FPS from actual frame timing
        fps_average = (frame_count - 1) / total_duration_pts if total_duration_pts > 0 else metadata['fps_nominal']
    else:
        fps_average = metadata['fps_nominal']

    # Compute inter-frame intervals for VFR characterization
    if frame_count >= 2:
        intervals = np.diff(pts_list)
        ifi_mean = float(np.mean(intervals))
        ifi_std = float(np.std(intervals))
        ifi_min = float(np.min(intervals))
        ifi_max = float(np.max(intervals))
    else:
        ifi_mean = ifi_std = ifi_min = ifi_max = 0.0

    # Assemble final metadata
    if crop_result is not None:
        metadata['crop_offset'] = {
            'x': crop_result['crop_x'],
            'y': crop_result['crop_y'],
            'w': crop_result['crop_w'],
            'h': crop_result['crop_h'],
        }

    metadata['frame_count'] = frame_count
    metadata['fps'] = fps_average
    metadata['vfr_stats'] = {
        'ifi_mean_sec': ifi_mean,
        'ifi_std_sec': ifi_std,
        'ifi_min_sec': ifi_min,
        'ifi_max_sec': ifi_max,
        'is_vfr': ifi_std > (ifi_mean * 0.01) if ifi_mean > 0 else False
    }

    # Save metadata JSON
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Metadata: {metadata_path}")
    print(f"    - Resolution: {metadata['width']}x{metadata['height']}")
    print(f"    - Frame count: {frame_count}")
    print(f"    - FPS (average): {fps_average:.4f}")
    print(f"    - FPS (nominal): {metadata['fps_nominal']:.4f}")
    print(f"    - Duration: {metadata['duration_sec']:.2f}s")
    print(f"    - VFR: {'Yes' if metadata['vfr_stats']['is_vfr'] else 'No'} "
          f"(IFI std={ifi_std*1000:.3f}ms)")

    # Save PTS as CSV: frame_idx, pts_seconds
    Path(pts_csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pts_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'pts_seconds'])
        for idx, pts in enumerate(pts_list):
            writer.writerow([idx, f"{pts:.6f}"])

    print(f"  [OK] PTS timestamps: {pts_csv_path} ({frame_count} frames)")


# =============================================================================
# ENTRY POINTS
# =============================================================================


def generate_cropped_representative_image(cropped_video: str, roi_json_path: str,
                                           output_image_path: str):
    """
    Generate a representative image from the cropped video with ROI overlay.

    Extracts a background frame from the cropped video and draws all ROI zones
    (floor, nest, loom) using the already-transformed coordinates. This produces
    a QC image that matches the coordinate space of the actual analysis.

    Args:
        cropped_video: Path to the cropped video
        roi_json_path: Path to ROI JSON (must already be in cropped space)
        output_image_path: Path to write the visualization JPEG
    """
    # Extract a representative frame from the cropped video
    cap = cv2.VideoCapture(cropped_video)
    if not cap.isOpened():
        print(f"  WARNING: Could not open cropped video for representative image")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames for a mean background (same approach as video_preprocessing)
    sampled_frames = []
    sample_interval = max(1, min(total_frames, 3600) // 30)
    frame_idx = 0

    while cap.isOpened() and frame_idx < min(3600, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            sampled_frames.append(frame)
        frame_idx += 1

    cap.release()

    if len(sampled_frames) == 0:
        print(f"  WARNING: No frames sampled from cropped video")
        return

    background = np.mean(np.array(sampled_frames), axis=0).astype(np.uint8)
    print(f"  Background from cropped video: {background.shape} ({len(sampled_frames)} frames sampled)")

    # Load ROI data (already in cropped space)
    with open(roi_json_path, 'r') as f:
        roi_data = json.load(f)

    vis_image = background.copy()

    if 'regions' in roi_data:
        for region in roi_data['regions']:
            if 'floor' in region:
                floor_pts = np.array(region['floor'], dtype=np.int32)
                cv2.polylines(vis_image, [floor_pts], isClosed=True, color=(0, 255, 0), thickness=3)
                for i, pt in enumerate(floor_pts):
                    cv2.circle(vis_image, tuple(pt), 8, (0, 255, 0), -1)
                    label = ["TL", "TR", "BR", "BL"][i]
                    cv2.putText(vis_image, label, (pt[0]+12, pt[1]-12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if 'nest' in region:
                nest_pts = np.array(region['nest'], dtype=np.int32)
                cv2.polylines(vis_image, [nest_pts], isClosed=True, color=(255, 128, 0), thickness=2)
                for pt in nest_pts:
                    cv2.circle(vis_image, tuple(pt), 5, (255, 128, 0), -1)

            if 'loom' in region:
                loom_center = np.array(region['loom'][0], dtype=np.int32)
                loom_radius = int(region['loom'][1])
                cv2.circle(vis_image, tuple(loom_center), loom_radius, (0, 0, 255), 2)
                cv2.circle(vis_image, tuple(loom_center), 5, (0, 0, 255), -1)

    # Legend
    legend_y = 30
    for label, color in [("Floor (Green)", (0, 255, 0)),
                          ("Nest (Orange)", (255, 128, 0)),
                          ("Loom (Red)", (0, 0, 255))]:
        cv2.putText(vis_image, label, (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        legend_y += 30

    # Note coordinate space
    cv2.putText(vis_image, "Cropped video space", (10, background.shape[0] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_image_path, vis_image)
    print(f"  [OK] Representative image (cropped): {output_image_path}")


def transform_roi_to_cropped_space(roi_json_input: str, roi_json_output: str, crop_result: dict):
    """
    Transform ROI JSON coordinates from original video space to cropped-video space.

    SLEAP runs on the cropped video, so body-part coordinates have (0,0) at
    the crop's top-left corner. The ROI JSON from video_preprocessing has
    coordinates in the original (uncropped) video space. This function
    subtracts the crop origin (crop_x, crop_y) from all spatial coordinates
    in the ROI JSON so downstream stages (roi_feature_extraction, feature_analysis)
    can compare body parts and ROI zones directly without offset correction.

    Args:
        roi_json_input: Path to input ROI JSON file (original video space)
        roi_json_output: Path to output ROI JSON file (cropped video space)
        crop_result: Dict from calculate_crop_params with crop_x, crop_y
    """
    with open(roi_json_input, 'r') as f:
        roi_data = json.load(f)

    offset_x = crop_result['crop_x']
    offset_y = crop_result['crop_y']
    print(f"  Crop offset: ({offset_x}, {offset_y})")

    if 'regions' not in roi_data:
        print("  WARNING: No 'regions' key in ROI JSON, skipping transform")
        # Still write output file even if empty
        with open(roi_json_output, 'w') as f:
            json.dump(roi_data, f, indent=2)
        return

    for region in roi_data['regions']:
        # Transform floor polygon
        if 'floor' in region:
            floor = np.array(region['floor'])
            floor[:, 0] -= offset_x
            floor[:, 1] -= offset_y
            region['floor'] = floor.tolist()
            print(f"  Transformed floor polygon ({len(floor)} vertices)")

        # Transform nest polygon
        if 'nest' in region:
            nest = np.array(region['nest'])
            nest[:, 0] -= offset_x
            nest[:, 1] -= offset_y
            region['nest'] = nest.tolist()
            print(f"  Transformed nest polygon ({len(nest)} vertices)")

        # Transform loom circle center (radius is unaffected)
        if 'loom' in region:
            loom_center = np.array(region['loom'][0])
            loom_center[0] -= offset_x
            loom_center[1] -= offset_y
            region['loom'][0] = loom_center.tolist()
            print(f"  Transformed loom center to ({loom_center[0]:.1f}, {loom_center[1]:.1f})")

    # Update metadata to record the transform
    if 'metadata' in roi_data:
        roi_data['metadata']['coordinate_space'] = 'cropped'
        roi_data['metadata']['crop_offset_applied'] = {
            'x': offset_x,
            'y': offset_y
        }

    # Write to new output file
    Path(roi_json_output).parent.mkdir(parents=True, exist_ok=True)
    with open(roi_json_output, 'w') as f:
        json.dump(roi_data, f, indent=2)

    print(f"  [OK] Transformed ROI JSON written to: {roi_json_output}")

def main():
    """Main entry point for command line or Snakemake."""
    try:
        # Snakemake integration
        input_video = snakemake.input.video
        roi_json_input = snakemake.input.roi_data  # Original space
        output_video = snakemake.output.cropped_video
        metadata_path = snakemake.output.metadata
        pts_csv_path = snakemake.output.pts_csv
        roi_json_output = snakemake.output.roi_data_cropped  # Cropped space
        representative_image = snakemake.output.representative_image
        threads = snakemake.threads
    except NameError:
        # Command line usage
        if len(sys.argv) < 8:
            print("Usage: crop_video.py <input_video> <roi_json_input> <output_video> "
                  "<metadata_json> <pts_csv> <roi_json_output> <representative_image> [threads]")
            sys.exit(1)

        input_video = sys.argv[1]
        roi_json_input = sys.argv[2]
        output_video = sys.argv[3]
        metadata_path = sys.argv[4]
        pts_csv_path = sys.argv[5]
        roi_json_output = sys.argv[6]
        representative_image = sys.argv[7]
        threads = int(sys.argv[8]) if len(sys.argv) > 8 else 4

    print("=" * 70)
    print("VIDEO CROP + METADATA EXTRACTION")
    print("=" * 70)

    # Step 1: Crop video (reads from original ROI JSON)
    print("\n[1/4] Cropping video...")
    crop_result = crop_video(input_video, roi_json_input, output_video, threads)

    # Step 2: Probe cropped video for metadata + PTS
    print("\n[2/4] Extracting metadata and frame timestamps...")
    probe_cropped_video(output_video, metadata_path, pts_csv_path, crop_result)

    # Step 3: Transform ROI coordinates from original to cropped video space (NEW OUTPUT FILE)
    print("\n[3/4] Transforming ROI coordinates to cropped video space...")
    transform_roi_to_cropped_space(roi_json_input, roi_json_output, crop_result)

    # Step 4: Generate representative image from cropped video with transformed ROIs
    print("\n[4/4] Generating representative image from cropped video...")
    generate_cropped_representative_image(output_video, roi_json_output, representative_image)

    print("\n" + "=" * 70)
    print("CROP + PROBE + ROI TRANSFORM + QC IMAGE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()