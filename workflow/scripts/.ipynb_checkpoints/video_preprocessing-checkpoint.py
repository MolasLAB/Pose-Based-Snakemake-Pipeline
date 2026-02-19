#!/usr/bin/env python3
"""
Video Preprocessing Script for Behavioral Pipeline
Combines ROI detection, metadata extraction, and background image generation.

This script:
1. Extracts video metadata (fps, duration, resolution)
2. Detects behavioral arena ROI using robust negative space detection
3. Generates a representative background image via median/mean filtering

Author: June (Molas Lab)
Integrated for Snakemake pipeline
"""

import os
import sys
import json
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add library path for robust_roi_detection
sys.path.append(str(Path(__file__).parent.parent.parent / 'lib'))
from robust_roi_detection import TrapBoxDetector


# =============================================================================
# VIDEO METADATA EXTRACTION
# =============================================================================

def ffmpeg_frame_probe(video_path: str) -> Tuple[float, float]:
    """
    Probe a video file to get its frames per second (fps) and duration using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        tuple: (fps, duration) where fps is frames per second and duration is in seconds
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse output
        lines = result.stdout.splitlines()
        fps_str = lines[0].strip()
        duration_str = lines[1].strip() if len(lines) > 1 else "0"
        
        # Parse FPS (may be fraction like "30000/1001")
        if '/' in fps_str:
            numerator, denominator = fps_str.split('/')
            fps = float(numerator) / float(denominator)
        else:
            fps = float(fps_str)
        
        duration = float(duration_str)
        return fps, duration
        
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"Warning: ffprobe failed ({e}), using fallback values")
        return 29.595, 0.0  # Default fallback


def extract_metadata(video_path: str) -> Dict:
    """
    Extract comprehensive metadata from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Metadata including fps, frame_count, width, height, duration_sec
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get basic properties from OpenCV
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_cv = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    # Get accurate FPS from ffprobe
    fps_probe, duration_probe = ffmpeg_frame_probe(video_path)
    
    # Use ffprobe values with OpenCV fallback
    fps = fps_probe if fps_probe > 0 else fps_cv
    duration = duration_probe if duration_probe > 0 else (frame_count / fps if fps > 0 else 0)
    
    metadata = {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration_sec': duration,
        'video_path': video_path
    }
    
    return metadata

#def exact_time_to_frame(video_path: str): -> Dict 
    # """
    # Take exact time_stamps and find the nearest frame (and error) to that timestamp in the video.
    # Potentially Useful for things like photometry where this level of precision may be necessary for alignment
    # """

# =============================================================================
# BACKGROUND IMAGE GENERATION
# =============================================================================

def generate_background(video_path: str, frame_span: int = 3600, 
                       use_median: bool = False) -> np.ndarray:
    """
    Generate representative background image by sampling frames across video.
    
    Uses mean by default (faster) with median as option (more robust to moving objects).
    
    Args:
        video_path: Path to video file
        frame_span: Maximum number of frames to sample from (default 3600 = ~2min @ 30fps)
        use_median: Use median instead of mean (slower but more robust)
        
    Returns:
        np.ndarray: Background image (H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps, duration = ffmpeg_frame_probe(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine sampling strategy
    if total_frames < frame_span:
        print(f"Video ({total_frames} frames) shorter than frame_span ({frame_span})")
        sample_interval = max(1, total_frames // 30)  # Sample ~30 frames
    else:
        sample_interval = 30  # Sample every 30 frames
    
    # Sample frames
    sampled_frames = []
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < min(frame_span, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            sampled_frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    if len(sampled_frames) == 0:
        raise RuntimeError("No frames sampled from video")
    
    print(f"Sampled {len(sampled_frames)} frames for background generation")
    
    # Compute background
    frames_array = np.array(sampled_frames)
    
    if use_median:
        background = np.median(frames_array, axis=0).astype(np.uint8)
    else:
        background = np.mean(frames_array, axis=0).astype(np.uint8)
    
    return background


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main(snakemake):
    """
    Main workflow for Snakemake integration.
    
    Inputs:
        snakemake.input.video: Path to input video
        
    Outputs:
        snakemake.output.metadata: JSON file with video metadata
        snakemake.output.roi_data: JSON file with ROI detection results
        snakemake.output.representative_image: JPEG background image
        
    Params:
        snakemake.params.frame_span: Number of frames to sample for background
        snakemake.params.min_area_ratio: Minimum ROI area as fraction of image
        snakemake.params.canny_low: Canny edge detection low threshold
        snakemake.params.canny_high: Canny edge detection high threshold
    """
    
    print("="*70)
    print("VIDEO PREPROCESSING PIPELINE")
    print("="*70)
    
    video_path = snakemake.input.video
    
    # Step 1: Extract metadata
    print("\n[1/3] Extracting video metadata...")
    metadata = extract_metadata(video_path)
    print(f"  - Resolution: {metadata['width']}x{metadata['height']}")
    print(f"  - FPS: {metadata['fps']:.3f}")
    print(f"  - Duration: {metadata['duration_sec']:.2f}s")
    print(f"  - Frames: {metadata['frame_count']}")
    
    # Step 2: Generate background image
    print("\n[2/3] Generating representative background image...")
    background = generate_background(
        video_path,
        frame_span=snakemake.params.frame_span,
        use_median=False  # Mean is faster and works well
    )
    print(f"  - Background shape: {background.shape}")
    
    # Step 3: Detect ROI
    print("\n[3/3] Detecting behavioral arena ROI...")
    detector = TrapBoxDetector(
        min_area_ratio=snakemake.params.min_area_ratio,
        canny_low=snakemake.params.canny_low,
        canny_high=snakemake.params.canny_high
    )
    
    roi_result = detector.detect_from_image(background)
    
    if roi_result is None:
        print("  ⚠ WARNING: ROI detection failed, using full frame")
        # Fallback to full frame
        h, w = background.shape[:2]
        roi_data = {
            'corners': [[0, 0], [w, 0], [w, h], [0, h]],
            'center': [w/2, h/2],
            'width': w,
            'height': h,
            'angle': 0,
            'confidence': 0.0,
            'method': 'fallback_full_frame'
        }
    else:
        print(f"  - Confidence: {roi_result.confidence:.3f}")
        print(f"  - Method: {roi_result.method}")
        print(f"  - Dimensions: {roi_result.width:.1f} x {roi_result.height:.1f}")
        roi_data = {
            'corners': roi_result.corners.tolist(),
            'center': roi_result.center,
            'width': float(roi_result.width),
            'height': float(roi_result.height),
            'angle': float(roi_result.angle),
            'confidence': float(roi_result.confidence),
            'method': roi_result.method
        }
    
    # Step 4: Save outputs
    print("\nSaving outputs...")
    
    # Save metadata
    with open(snakemake.output.metadata, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata: {snakemake.output.metadata}")
    
    # Save ROI data
    with open(snakemake.output.roi_data, 'w') as f:
        json.dump(roi_data, f, indent=2)
    print(f"  ✓ ROI data: {snakemake.output.roi_data}")
    
    # Save background image (cropped to ROI if detected)
    # TODO: Consider cropping the background to ROI here
    cv2.imwrite(snakemake.output.representative_image, background)
    print(f"  ✓ Background: {snakemake.output.representative_image}")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    # When run via Snakemake, 'snakemake' object is automatically available
    main(snakemake)
