#!/usr/bin/env python3
"""
Video Preprocessing Script for Behavioral Pipeline
Handles ROI definition (manual or automatic) and background image generation.

This script:
1. Generates a representative background image via median/mean filtering (OpenCV only)
2. Defines behavioral arena ROI (manual GUI labeling or automatic detection)
3. Calculates nest and loom zones from the arena corners
4. Outputs ROI JSON and visualization image

Note: Video metadata extraction and frame timestamp probing are handled
downstream by crop_video.py (post-crop ffprobe), keeping this script
focused on the interactive/sequential ROI labeling phase.

Author: June (Molas Lab)
Integrated for Snakemake pipeline
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Add library path for robust_roi_detection
sys.path.append(str(Path(__file__).parent.parent.parent / 'lib'))
from robust_roi_detection import TrapBoxDetector


# =============================================================================
# BACKGROUND IMAGE GENERATION (OpenCV only no ffmpeg)
# =============================================================================

def generate_background(video_path: str, frame_span: int = 3600,
                       use_median: bool = False) -> np.ndarray:
    """
    Generate representative background image by sampling frames across video.

    Uses mean by default (faster) with median as option (more robust to moving objects).
    Uses OpenCV only  no ffmpeg/ffprobe dependency.

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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine sampling strategy
    if total_frames < frame_span:
        print(f"  Video ({total_frames} frames) shorter than frame_span ({frame_span})")
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

    print(f"  Sampled {len(sampled_frames)} frames for background generation")

    # Compute background
    frames_array = np.array(sampled_frames)

    if use_median:
        background = np.median(frames_array, axis=0).astype(np.uint8)
    else:
        background = np.mean(frames_array, axis=0).astype(np.uint8)

    return background


# =============================================================================
# NEST AND LOOM ZONE CALCULATIONS
# =============================================================================

def nest_calculator(f_pts: np.ndarray, percent_of_perspective: float = 0.1955,
                   buffer: float = 80) -> np.ndarray:
    """
    Calculate nest zone from floor corners using original algorithm.

    This is a hard coded calculation of where the nest should be given the floor's corners.
    The rectangle's points are defined as:
    0-1
    | |
    3-2

    Args:
        f_pts: Floor corners as 4x2 array [TL, TR, BR, BL] (CW from top-left, COCO standard)
        percent_of_perspective: Percentage down from top for nest placement (default 0.1955)
        buffer: Buffer distance from floor edges (default 80 pixels)

    Returns:
        np.ndarray: 4x2 array of nest corners [TL, TR, BR, BL] (CW from top-left)
    """
    nest_rect = np.zeros((4, 2))

    side_vector = f_pts[2] - f_pts[3]
    side_vector = side_vector / np.linalg.norm(side_vector)
    up_vector = np.array([side_vector[1], -side_vector[0]])

    nest_rect[0] = (percent_of_perspective * f_pts[0] +
                    (1 - percent_of_perspective) * f_pts[3] -
                    4*side_vector * buffer)
    nest_rect[2] = f_pts[2] + 4 * buffer * side_vector - buffer * up_vector
    nest_rect[3] = f_pts[3] - 4 * buffer * side_vector - buffer * up_vector
    nest_rect[1] = (percent_of_perspective * f_pts[1] +
                    (1 - percent_of_perspective) * f_pts[2] +
                    4 * side_vector * buffer)

    return nest_rect


def loom_calculator(f_pts: np.ndarray, percent_of_perspective: float = 0.1506,
                   radius: float = 32) -> Tuple[np.ndarray, float]:
    """
    Calculate loom zone center and radius from floor corners.

    Args:
        f_pts: Floor corners as 4x2 array [TL, TR, BR, BL] (CW from top-left, COCO standard)
        percent_of_perspective: Percentage down from top for loom placement (default 0.1506)
        radius: Loom circle radius in pixels (default 32)

    Returns:
        tuple: (center as [x, y], radius)
    """
    loom_circle = np.zeros(2)

    down_vector = f_pts[3] - f_pts[0]  # Vector from TL (0) to BL (3)
    loom_circle = 0.5 * (f_pts[0] + f_pts[1]) + (down_vector * percent_of_perspective)

    return loom_circle, radius


# =============================================================================
# MANUAL ROI LABELING GUI
# =============================================================================

class ManualROILabeler:
    """
    Interactive GUI for manually labeling the behavioral arena floor.

    User can click to place 4 corners (top-left, then CW: top-right,
    bottom-right, bottom-left) and drag to adjust positions.

    COCO standard ordering: [TL, TR, BR, BL] (clockwise from top-left)

    Controls:
        - Left click: Place/select point
        - Drag: Move selected point
        - 'r': Reset all points
        - 'a': Accept current points (if all 4 are placed)
        - Enter: Same as 'a'
        - Esc: Cancel (use previous ROI if available, else error)
    """

    def __init__(self, image: np.ndarray, previous_roi: Optional[np.ndarray] = None):
        self.image = image.copy()
        self.display_image = None
        self.original_image = image.copy()

        self.points: List[Tuple[int, int]] = []
        if previous_roi is not None and len(previous_roi) == 4:
            self.points = [tuple(map(int, p)) for p in previous_roi]

        self.selected_point_idx: Optional[int] = None
        self.dragging = False

        self.point_radius = 8
        self.line_thickness = 2
        self.point_colors = [
            (0, 255, 0),    # Top-left: Green
            (255, 255, 0),  # Top-right: Cyan
            (0, 0, 255),    # Bottom-right: Red
            (255, 0, 0)     # Bottom-left: Blue
        ]
        self.line_color = (0, 255, 255)  # Yellow
        self.selected_color = (255, 255, 255)  # White

        self.window_name = "Manual ROI Labeling - Place 4 corners (Top-Left, then CW)"

    def _find_nearest_point(self, x: int, y: int, threshold: int = 15) -> Optional[int]:
        if not self.points:
            return None
        distances = [np.sqrt((px - x)**2 + (py - y)**2) for px, py in self.points]
        min_idx = np.argmin(distances)
        if distances[min_idx] <= threshold:
            return min_idx
        return None

    def _draw_overlay(self):
        self.display_image = self.original_image.copy()

        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            if len(self.points) == 4:
                cv2.polylines(self.display_image, [pts], isClosed=True,
                            color=self.line_color, thickness=self.line_thickness)
            else:
                cv2.polylines(self.display_image, [pts], isClosed=False,
                            color=self.line_color, thickness=self.line_thickness)

        for idx, (px, py) in enumerate(self.points):
            color = self.selected_color if idx == self.selected_point_idx else self.point_colors[idx]
            cv2.circle(self.display_image, (px, py), self.point_radius, color, 2)
            cv2.circle(self.display_image, (px, py), self.point_radius + 2, (255, 255, 255), 1)
            cv2.circle(self.display_image, (px, py), 2, color, -1)

            label = ["TL", "TR", "BR", "BL"][idx]
            cv2.putText(self.display_image, label, (px + 12, py - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        instructions = [
            f"Points: {len(self.points)}/4",
            "Click: Place/Select | Drag: Move | R: Reset",
            "A/Enter: Accept | Esc: Cancel"
        ]

        y_offset = 30
        for instruction in instructions:
            cv2.putText(self.display_image, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        if len(self.points) == 4:
            cv2.putText(self.display_image, "Press 'A' or Enter to accept!",
                       (10, self.display_image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nearest_idx = self._find_nearest_point(x, y)
            if nearest_idx is not None:
                self.selected_point_idx = nearest_idx
                self.dragging = True
            elif len(self.points) < 4:
                self.points.append((x, y))
                self.selected_point_idx = len(self.points) - 1
            self._draw_overlay()
            cv2.imshow(self.window_name, self.display_image)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_point_idx is not None:
                self.points[self.selected_point_idx] = (x, y)
                self._draw_overlay()
                cv2.imshow(self.window_name, self.display_image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def label(self) -> Optional[np.ndarray]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._draw_overlay()
        cv2.imshow(self.window_name, self.display_image)

        if len(self.points) == 4:
            print("\n  Previous ROI loaded. Press 'A' or Enter to accept, or adjust points.")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') or key == ord('R'):
                self.points = []
                self.selected_point_idx = None
                self.dragging = False
                print("  Points reset")
                self._draw_overlay()
                cv2.imshow(self.window_name, self.display_image)

            elif key == ord('a') or key == ord('A') or key == 13:
                if len(self.points) == 4:
                    cv2.destroyAllWindows()
                    return np.array(self.points, dtype=np.float32)
                else:
                    print(f"  Need 4 points, currently have {len(self.points)}")

            elif key == 27:
                cv2.destroyAllWindows()
                if len(self.points) == 4:
                    print("  Cancelled - using previous ROI")
                    return np.array(self.points, dtype=np.float32)
                else:
                    print("  Cancelled - no valid ROI")
                    return None


def load_previous_manual_roi(config_path: Optional[str] = None,
                             video_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Load previously saved manual ROI from config file."""
    if config_path is None or not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if video_name and video_name in config.get('manual_rois', {}):
            roi_data = config['manual_rois'][video_name]
            return np.array(roi_data, dtype=np.float32)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Warning: Could not load previous ROI from config: {e}")

    return None


def save_manual_roi_to_config(roi_corners: np.ndarray,
                              config_path: str,
                              video_name: str):
    """Save manual ROI to config file for future use."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    if 'manual_rois' not in config:
        config['manual_rois'] = {}

    config['manual_rois'][video_name] = roi_corners.tolist()

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Saved manual ROI to config: {config_path}")


def manual_roi_labeling(background: np.ndarray,
                       config_path: Optional[str] = None,
                       video_name: Optional[str] = None,
                       nest_percent: float = 0.1955,
                       nest_buffer: float = 40,
                       loom_percent: float = 0.1506,
                       loom_radius: float = 32) -> Dict:
    """
    Main function for manual ROI labeling with GUI.

    Args:
        background: Background image to label
        config_path: Optional path to config file for loading/saving ROIs
        video_name: Name of video (for config lookup)
        nest_percent: Percentage of perspective for nest calculation
        nest_buffer: Buffer distance for nest zone
        loom_percent: Percentage of perspective for loom calculation
        loom_radius: Radius of loom circle

    Returns:
        dict: Complete ROI data including floor, nest, and loom zones
    """
    print("\n[MANUAL ROI LABELING]")

    previous_roi = load_previous_manual_roi(config_path, video_name)
    if previous_roi is not None:
        print(f"  Loaded previous ROI for {video_name}")

    labeler = ManualROILabeler(background, previous_roi=previous_roi)
    roi_corners = labeler.label()

    if roi_corners is None:
        raise RuntimeError("Manual ROI labeling cancelled without valid ROI")

    if config_path and video_name:
        save_manual_roi_to_config(roi_corners, config_path, video_name)

    nest_corners = nest_calculator(roi_corners, nest_percent, nest_buffer)
    loom_center, loom_rad = loom_calculator(roi_corners, loom_percent, loom_radius)

    roi_data = {
        'regions': [
            {'floor': roi_corners.tolist()},
            {'nest': nest_corners.tolist()},
            {'loom': [loom_center.tolist(), loom_rad]}
        ]
    }

    center = roi_corners.mean(axis=0)
    width = (np.linalg.norm(roi_corners[1] - roi_corners[0]) +
             np.linalg.norm(roi_corners[2] - roi_corners[3])) / 2
    height = (np.linalg.norm(roi_corners[3] - roi_corners[0]) +
              np.linalg.norm(roi_corners[2] - roi_corners[1])) / 2
    top_edge = roi_corners[1] - roi_corners[0]
    angle = np.arctan2(top_edge[1], top_edge[0]) * 180 / np.pi

    roi_data['metadata'] = {
        'center': center.tolist(),
        'width': float(width),
        'height': float(height),
        'angle': float(angle),
        'confidence': 1.0,
        'method': 'manual_gui_labeling'
    }

    print(f"  [OK] Manual ROI labeled: {width:.1f} x {height:.1f} pixels")
    print(f"  [OK] Nest zone calculated: {len(nest_corners)} corners")
    print(f"  [OK] Loom zone calculated: center={loom_center}, radius={loom_rad}")

    return roi_data


def visualize_roi_zones(background: np.ndarray, roi_data: Dict) -> np.ndarray:
    """Create visualization of all ROI zones on background image."""
    vis_image = background.copy()

    floor_pts = np.array(roi_data['regions'][0]['floor'], dtype=np.int32)
    nest_pts = np.array(roi_data['regions'][1]['nest'], dtype=np.int32)
    loom_center = np.array(roi_data['regions'][2]['loom'][0], dtype=np.int32)
    loom_radius = int(roi_data['regions'][2]['loom'][1])

    cv2.polylines(vis_image, [floor_pts], isClosed=True, color=(0, 255, 0), thickness=3)
    for i, pt in enumerate(floor_pts):
        cv2.circle(vis_image, tuple(pt), 8, (0, 255, 0), -1)
        label = ["TL", "TR", "BR", "BL"][i]
        cv2.putText(vis_image, label, (pt[0]+12, pt[1]-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.polylines(vis_image, [nest_pts], isClosed=True, color=(255, 128, 0), thickness=2)
    for pt in nest_pts:
        cv2.circle(vis_image, tuple(pt), 5, (255, 128, 0), -1)

    cv2.circle(vis_image, tuple(loom_center), loom_radius, (0, 0, 255), 2)
    cv2.circle(vis_image, tuple(loom_center), 5, (0, 0, 255), -1)

    legend_y = 30
    cv2.putText(vis_image, "Floor (Green)", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    legend_y += 30
    cv2.putText(vis_image, "Nest (Orange)", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    legend_y += 30
    cv2.putText(vis_image, "Loom (Red)", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return vis_image


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main(snakemake):
    """
    Main workflow for Snakemake integration.

    Inputs:
        snakemake.input.video: Path to input video

    Outputs:
        snakemake.output.roi_data: JSON file with ROI detection results (floor, nest, loom)
        snakemake.output.representative_image: JPEG visualization with all ROI zones

    Note:
        Video metadata (fps, resolution, PTS timestamps) is now extracted
        downstream by crop_video.py after cropping. This script focuses
        exclusively on the interactive ROI labeling phase.
    """

    print("="*70)
    print("VIDEO PREPROCESSING ROI LABELING")
    print("="*70)

    video_path = snakemake.input.video
    video_name = Path(video_path).stem

    use_manual_roi = getattr(snakemake.params, 'use_manual_roi', False)
    manual_roi_config = getattr(snakemake.params, 'manual_roi_config', None)

    nest_percent = getattr(snakemake.params, 'nest_percent', 0.1955)
    nest_buffer = getattr(snakemake.params, 'nest_buffer', 40)
    loom_percent = getattr(snakemake.params, 'loom_percent', 0.1506)
    loom_radius = getattr(snakemake.params, 'loom_radius', 32)

    # Step 1: Generate background image (OpenCV only)
    print("\n[1/3] Generating representative background image...")
    background = generate_background(
        video_path,
        frame_span=snakemake.params.frame_span,
        use_median=False
    )
    print(f"  - Background shape: {background.shape}")

    # Step 2: Detect or manually label ROI
    print("\n[2/3] Determining behavioral arena ROI...")

    if use_manual_roi:
        print("  MODE: Manual GUI Labeling")
        try:
            roi_data = manual_roi_labeling(
                background,
                config_path=manual_roi_config,
                video_name=video_name,
                nest_percent=nest_percent,
                nest_buffer=nest_buffer,
                loom_percent=loom_percent,
                loom_radius=loom_radius
            )
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            raise
    else:
        print("  MODE: Automatic Detection")

        detector = TrapBoxDetector(
            min_area_ratio=snakemake.params.min_area_ratio
        )

        roi_result = detector.detect_from_image(background)

        if roi_result is None:
            print("  [WARNING] ROI detection failed, using full frame")
            h, w = background.shape[:2]
            floor_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            confidence = 0.0
            method = 'fallback_full_frame'
        else:
            print(f"  - Confidence: {roi_result.confidence:.3f}")
            print(f"  - Method: {roi_result.method}")
            print(f"  - Dimensions: {roi_result.width:.1f} x {roi_result.height:.1f}")
            floor_corners = np.array(roi_result.corners, dtype=np.float32)
            confidence = float(roi_result.confidence)
            method = roi_result.method

        nest_corners = nest_calculator(floor_corners, nest_percent, nest_buffer)
        loom_center, loom_rad = loom_calculator(floor_corners, loom_percent, loom_radius)

        roi_data = {
            'regions': [
                {'floor': floor_corners.tolist()},
                {'nest': nest_corners.tolist()},
                {'loom': [loom_center.tolist(), loom_rad]}
            ]
        }

        center = floor_corners.mean(axis=0)
        width = (np.linalg.norm(floor_corners[1] - floor_corners[0]) +
                 np.linalg.norm(floor_corners[2] - floor_corners[3])) / 2
        height = (np.linalg.norm(floor_corners[3] - floor_corners[0]) +
                  np.linalg.norm(floor_corners[2] - floor_corners[1])) / 2
        top_edge = floor_corners[1] - floor_corners[0]
        angle = np.arctan2(top_edge[1], top_edge[0]) * 180 / np.pi

        roi_data['metadata'] = {
            'center': center.tolist(),
            'width': float(width),
            'height': float(height),
            'angle': float(angle),
            'confidence': confidence,
            'method': method
        }

        print(f"  [OK] Floor ROI: {width:.1f} x {height:.1f} pixels")
        print(f"  [OK] Nest zone calculated: {len(nest_corners)} corners")
        print(f"  [OK] Loom zone calculated: center={loom_center}, radius={loom_rad}")

    # Step 3: Save ROI data
    # Note: Representative image is now generated by crop_video.py using
    # the cropped video and transformed ROI coordinates, so the QC image
    # matches the coordinate space used by SLEAP and downstream analysis.
    print("\n[3/3] Saving ROI data...")

    with open(snakemake.output.roi_data, 'w') as f:
        json.dump(roi_data, f, indent=2)
    print(f"  [OK] ROI data: {snakemake.output.roi_data}")

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main(snakemake)