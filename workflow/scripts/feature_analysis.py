#!/usr/bin/env python3
"""
Feature Analysis Script for Behavioral Pipeline

Event-oriented behavioral analysis that processes extracted features around
loom events. Calculates metrics like freezing, latency to nest, velocity,
and social interaction features.

Supports both single and paired animal experiments.

Integrates with:
- Pipeline-generated ROI JSON (nest/loom zones)
- Pipeline metadata JSON (fps, resolution)
- Events CSV from event_convertor (frame-accurate event timing via PTS)
- Feature extraction output (features_with_roi.csv)

Author: June (Molas Lab)
Adapted for Snakemake pipeline integration
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import scipy as sp
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from shapely import Point, Polygon, points
from shapely import distance as shapely_distance

# Import feature configurations
from analysis_config import build_calculated_features
from analysis_config_single import build_calculated_features_single


# =============================================================================
# EVENT LOADING (replaces direct scoring sheet access)
# =============================================================================

def load_loom_frames_from_events(events_csv: str) -> List[int]:
    """
    Load loom event frame indices from the events CSV produced by event_convertor.

    The events CSV has columns:
        event_id, event_type, time_start, time_end,
        frame_start, frame_end, frame_error_start, frame_error_end

    This extracts frame_start for all events with event_type == 'loom',
    ordered by time_start (which the events CSV is already sorted by).

    Args:
        events_csv: Path to events CSV from event_convertor

    Returns:
        List of frame indices for loom events, ordered chronologically
    """
    df = pd.read_csv(events_csv)

    # Filter to loom events
    loom_events = df[df['event_type'] == 'loom'].sort_values('time_start')

    loom_frames = loom_events['frame_start'].astype(int).tolist()

    if len(loom_frames) == 0:
        print("  Warning: No loom events found in events CSV")

    return loom_frames


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class AnimalBehaviorAnalyzer:
    """
    Event-oriented behavioral analysis around loom events.

    Calculates features like:
    - Freezing behavior (start, duration, percentage)
    - Latency to nest entry
    - Time in nest
    - Head angle to loom
    - Velocity metrics
    - Social interaction features (for paired animals)

    Supports both single and paired animal experiments. Column naming
    follows the convention from roi_feature_extraction.py:
        ROI columns: {ROI_Name}_Animal_{id}_{bodypart}_in_zone
        Facing columns: {ROI_Name}_Animal_{id}_facing
        Pose columns: {bodypart}_{id}_x, {bodypart}_{id}_y
        Movement columns: movement_mouse_{id}_{bodypart}_mean_{window}
    """

    def __init__(self,
                 features_csv: str,
                 roi_json: str,
                 metadata_json: str,
                 events_csv: str,
                 animal_ids: tuple,
                 output_directory: str,
                 pixels_per_cm: float = 32,
                 start_velocity: float = 4,
                 end_velocity: float = 3):
        """
        Initialize the analyzer.

        Args:
            features_csv: Path to features_with_roi.csv from pipeline
            roi_json: Path to ROI JSON from pipeline
            metadata_json: Path to metadata JSON from pipeline
            events_csv: Path to events CSV from event_convertor
            animal_ids: Tuple of animal IDs
            output_directory: Directory for output files
            pixels_per_cm: Pixel to cm conversion
            start_velocity: Velocity threshold to start freezing (cm/s)
            end_velocity: Velocity threshold to end freezing (cm/s)
        """
        # Load metadata
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
        fps = metadata.get('fps', 29.595)

        # Load loom times from events CSV (frame-accurate via PTS)
        self.loom_times = load_loom_frames_from_events(events_csv)

        # Load ROI data
        self.roi_data = self._load_roi_data(roi_json)

        # Configuration
        self.pair = len(animal_ids) == 2
        self.fps = fps
        self.animal_ids = animal_ids
        self.output_directory = output_directory
        self.pixels_per_cm = pixels_per_cm
        self.start_velocity = start_velocity
        self.end_velocity = end_velocity

        # Velocity thresholds in pixels/frame
        self.VelThreshLow = self.end_velocity * self.pixels_per_cm / self.fps
        self.VelThreshHigh = self.start_velocity * self.pixels_per_cm / self.fps

        # Interpolation detection parameters
        self.interpolation_tolerance = 0.02
        self.interpolation_limit = 8

        # Load feature data
        self.df = pd.read_csv(features_csv)
        self.dataframe_length = len(self.df)

        #random edit
        # Initialize timepoints tracking
        self.loom_session_and_end = self.loom_times[:] + [self.dataframe_length]

        self.timepoints = [{
            "loomtime": self.loom_times[:] + [self.dataframe_length],
            "latency": [-1 for _ in range(len(self.loom_times))],
            "time_in_nest": [-1 for _ in range(len(self.loom_times))]
        } for _ in range(len(self.animal_ids))]
        print(self.timepoints[0]["loomtime"])
        # Build feature configuration based on experiment type
        if self.pair:
            self.CalculatedFeatures = build_calculated_features(self.fps, self.pixels_per_cm)
        else:
            self.CalculatedFeatures = build_calculated_features_single(self.fps, self.pixels_per_cm)

        # Detect ROI column naming convention from the loaded data
        self._detect_roi_column_format()

        print(f"  Loaded {self.dataframe_length} frames")
        print(f"  Found {len(self.loom_times)} loom events")
        print(f"  Animals: {self.animal_ids}")
        print(f"  Mode: {'Paired' if self.pair else 'Single'}")

    def _load_roi_data(self, roi_json: str) -> dict:
        """Load ROI data from pipeline JSON."""
        with open(roi_json, 'r') as f:
            data = json.load(f)

        roi_data = {
            'nest_polygon': None,
            'loom_circle': None
        }

        if 'regions' in data:
            for region in data['regions']:
                if 'nest' in region:
                    roi_data['nest_polygon'] = np.array(region['nest'], dtype=np.float32)
                if 'loom' in region:
                    roi_data['loom_circle'] = {
                        'center': np.array(region['loom'][0], dtype=np.float32),
                        'radius': float(region['loom'][1])
                    }

        return roi_data

    def _detect_roi_column_format(self):
        """
        Detect the ROI column naming convention used in the features CSV.
        """
        sample_cols = [c for c in self.df.columns if 'in_zone' in c or 'in zone' in c]
        
        if not sample_cols:
            print("  Warning: No ROI columns found!")
            # Use default underscore format
            self._roi_col_sep = '_'
            self._roi_zone_suffix = '_in_zone'
            return
        
        # Actually check the format of existing columns
        sample_col = sample_cols[0]
        
        if '_in_zone' in sample_col:
            # Underscore format (current standard)
            self._roi_col_sep = '_'
            self._roi_zone_suffix = '_in_zone'
            self._roi_dist_suffix = '_distance'
            self._roi_facing_suffix = '_facing'
        elif ' in zone' in sample_col:
            # Space format (legacy/alternate)
            self._roi_col_sep = ' '
            self._roi_zone_suffix = ' in zone'
            self._roi_dist_suffix = ' distance'
            self._roi_facing_suffix = ' facing'

    def _roi_zone_col(self, roi_name: str, animal_index: int, bodypart: str) -> str:
        sep = self._roi_col_sep
        return f"{roi_name}{sep}Animal_{animal_index}{sep}{bodypart}_{animal_index}{self._roi_zone_suffix}"

    def _roi_facing_col(self, roi_name: str, animal_index: int) -> str:
        sep = self._roi_col_sep
        return f"{roi_name}{sep}Animal_{animal_index}{self._roi_facing_suffix}"

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def feature_grab(self, feature_name: str) -> pd.Series:
        """Get a feature column from the dataframe."""
        if feature_name not in self.df.columns:
            raise KeyError(
                f"Column '{feature_name}' not found in dataframe. "
                f"Available columns with similar prefix: "
                f"{[c for c in self.df.columns if c.startswith(feature_name[:15])]}"
            )
        return self.df[feature_name]

    def window_cropper(self, animal_index: int, loom_index: int,
                       frame_start_point: str, frame_end_offset: int,
                       frame_start_offset: int = 0, frame_end_point: int = 0,
                       frame_limit: int = 0) -> List[int]:
        """
        Calculate start and end frames for a window around an event.

        Args:
            animal_index: 1-based animal index
            loom_index: Index of the loom event
            frame_start_point: Timepoint reference ("loomtime", "latency", etc.)
            frame_end_offset: Offset from end point
            frame_start_offset: Offset from start point
            frame_end_point: Optional different end reference
            frame_limit: Maximum window size

        Returns:
            [start_frame, end_frame] or [0, 0] if invalid
        """
        if self.timepoints[animal_index-1][frame_start_point][loom_index] == -1:
            return [0, 0]

        if frame_end_point == 0:
            frame_end_point = frame_start_point
        elif self.timepoints[animal_index-1][frame_end_point][loom_index] == -1:
            return [0, 0]

        frame_start = self.timepoints[animal_index-1][frame_start_point][loom_index] + frame_start_offset
        frame_end = self.timepoints[animal_index-1][frame_end_point][loom_index] + frame_end_offset

        if frame_limit != 0 and (frame_end - frame_start) > frame_limit:
            frame_end = frame_start + frame_limit

        if frame_end >= self.dataframe_length:
            frame_end = self.dataframe_length

        return [frame_start, frame_end]

    def window_cropper_both(self, loom_index: int, **window_kwargs) -> List[int]:
        """Get intersection of windows for both animals (pair mode only)."""
        if not self.pair:
            return [0, 0]

        window_1 = self.window_cropper(animal_index=1, loom_index=loom_index, **window_kwargs)
        window_2 = self.window_cropper(animal_index=2, loom_index=loom_index, **window_kwargs)

        if window_1 == [0, 0] or window_2 == [0, 0]:
            return [0, 0]

        intersection_start = max(window_1[0], window_2[0])
        intersection_end = min(window_1[1], window_2[1])

        if intersection_start >= intersection_end:
            return [0, 0]

        return [intersection_start, intersection_end]

    # =========================================================================
    # NEST DETECTION
    # =========================================================================

    def AnimalInNest(self, animal_index: int) -> np.ndarray:
        """Detect when all body parts of an animal are inside the nest."""
        parts = ['nose', 'ear_left', 'ear_right', 'head', 'body_center', 'tail_base']
        col_names = [self._roi_zone_col('Nest', animal_index, part) for part in parts]

        roi_data = self.feature_grab(col_names[0]).to_numpy()
        for col in col_names[1:]:
            roi_data = np.multiply(self.feature_grab(col).to_numpy(), roi_data)

        return roi_data

    def AnimalOutNest(self, animal_index: int) -> np.ndarray:
        """Detect when any body part of an animal is outside the nest."""
        parts = ['nose', 'ear_left', 'ear_right', 'head', 'body_center']
        col_names = [self._roi_zone_col('Nest', animal_index, part) for part in parts]

        roi_data = np.multiply(np.add(self.feature_grab(col_names[0]).to_numpy(), -1), -1)
        for col in col_names[1:]:
            roi_data = np.multiply(np.multiply(np.add(self.feature_grab(col), -1), -1).to_numpy(), roi_data)

        return np.add(np.multiply(roi_data, -1), 1)

    # =========================================================================
    # BEHAVIORAL METRICS
    # =========================================================================

    def AnimalFreezing(self, animal_index: int, threshold: float) -> np.ndarray:
        """Detect freezing based on velocity threshold."""
        feature = f'movement_mouse_{animal_index}_tail_base_mean_6'
        return np.less(self.feature_grab(feature).to_numpy(), threshold)

    def AnimalVelocity(self, animal_index: int) -> np.ndarray:
        """Calculate average velocity across body parts."""
        features = [
            f'movement_mouse_{animal_index}_nose_mean_6',
            f'movement_mouse_{animal_index}_tail_base_mean_6',
            f'movement_mouse_{animal_index}_ear_left_mean_6',
            f'movement_mouse_{animal_index}_ear_right_mean_6'
        ]
        velocity = self.feature_grab(features[0]).to_numpy()
        for f in features[1:]:
            velocity = np.add(self.feature_grab(f).to_numpy(), velocity)
        return np.multiply(velocity, 1 / len(features))

    def LatencyToNest(self, animal_index: int, loom_index: int) -> float:
        """Calculate frames until animal enters nest after loom."""
        loomtime = self.loom_session_and_end[loom_index]
        nexttime = self.loom_session_and_end[loom_index + 1]

        if f'animal_in_nest_{animal_index}' not in self.df:
            self.df[f'animal_in_nest_{animal_index}'] = self.AnimalInNest(animal_index)

        framecount = 0
        for i in range(len(self.df[f'animal_in_nest_{animal_index}']) - loomtime):
            if i + loomtime >= nexttime:
                self.timepoints[animal_index-1]["latency"][loom_index] = -1
                return -self.fps
            if self.df[f'animal_in_nest_{animal_index}'][i + loomtime] == 1:
                framecount += 1
            if framecount >= 5:
                nest_entry_frame = loomtime + (i - 4)
                self.timepoints[animal_index-1]["latency"][loom_index] = nest_entry_frame
                return i - 4

        self.timepoints[animal_index-1]["latency"][loom_index] = -1
        return -self.fps

    def TimeInNest(self, animal_index: int, loom_index: int) -> float:
        """Calculate time animal stays in nest after entry."""
        loomtime = self.loom_session_and_end[loom_index]

        if f'animal_out_nest_{animal_index}' not in self.df:
            self.df[f'animal_out_nest_{animal_index}'] = self.AnimalOutNest(animal_index)

        first_frame = self.LatencyToNest(animal_index, loom_index)
        if first_frame is None or first_frame < 0:
            self.timepoints[animal_index-1]["time_in_nest"][loom_index] = -1
            return -self.fps

        for i in range(self.dataframe_length - int(first_frame) - loomtime):
            if self.df[f'animal_out_nest_{animal_index}'][i + int(first_frame) + loomtime] == 0:
                self.timepoints[animal_index-1]["time_in_nest"][loom_index] = loomtime + int(first_frame) + i
                return i

        self.timepoints[animal_index-1]["time_in_nest"][loom_index] = -1
        return -self.fps

    def FreezingAfterLoom(self, animal_index: int, loom_index: int) -> Tuple:
        """Detect freezing start and duration after loom."""
        loomtime = self.loom_session_and_end[loom_index]
        low_thresh = self.AnimalFreezing(animal_index, self.VelThreshLow)
        high_thresh = self.AnimalFreezing(animal_index, self.VelThreshHigh)

        nest_entry = self.LatencyToNest(animal_index, loom_index)
        freeze_start = -1
        framecount = 0

        for i in range(len(low_thresh) - loomtime):
            if low_thresh[i + loomtime]:
                framecount += 1
            if i >= nest_entry and nest_entry > 0:
                return [-1, -1, high_thresh]
            if framecount >= 35:
                freeze_start = i - 34
                break

        if freeze_start == -1:
            return [-1, -1, -1]

        framecount = 0
        for i in range(len(high_thresh) - loomtime - freeze_start):
            if not high_thresh[i + loomtime + freeze_start]:
                framecount += 1
            if framecount >= 5:
                return [freeze_start, i - 5, high_thresh]

        return [freeze_start, len(low_thresh) - loomtime - freeze_start, high_thresh]

    def FreezePercentageAfterLoom(self, animal_index: int, loom_index: int, **kwargs) -> float:
        """Calculate freeze percentage over a window."""
        start_frame, end_frame = self.window_cropper(animal_index, loom_index, **kwargs["window"])
        _, _, freeze_frames = self.FreezingAfterLoom(animal_index, loom_index)

        if isinstance(freeze_frames, int) and freeze_frames == -1:
            return 0

        window_length = end_frame - start_frame
        if window_length <= 0:
            return 0

        return 100 * (freeze_frames[start_frame:end_frame].sum() / window_length)

    # =========================================================================
    # ROI-BASED FEATURES
    # =========================================================================

    def ROIPolygonDistance(self, animal_index: int) -> np.ndarray:
        """Calculate distance from each body part to the nest polygon."""
        if self.roi_data['nest_polygon'] is None:
            return np.zeros((self.dataframe_length, 6))

        poly = Polygon(self.roi_data['nest_polygon'])
        parts = ['nose', 'ear_left', 'ear_right', 'head', 'body_center', 'tail_base']

        pose_features = [f'{part}_{animal_index}_{axis}' for part in parts for axis in ['x', 'y']]

        pose_data = np.stack([self.feature_grab(f).to_numpy() for f in pose_features], axis=1)
        skeleton_points = pose_data.reshape((-1, 6, 2))

        x_coords = skeleton_points[:, :, 0].ravel()
        y_coords = skeleton_points[:, :, 1].ravel()
        point_array = points(x_coords, y_coords)

        all_distances = shapely_distance(poly, point_array)
        distance_array = all_distances.reshape(-1, 6)

        for i, part in enumerate(parts):
            col_name = f'roi_distance_{part}_animal_{animal_index}'
            self.df[col_name] = distance_array[:, i]

        return distance_array

    def ROIPolygonVelocity(self, animal_index: int) -> np.ndarray:
        """Calculate velocity towards/away from nest polygon."""
        parts = ['nose', 'ear_left', 'ear_right', 'head', 'body_center', 'tail_base']
        distance_array = self.ROIPolygonDistance(animal_index)

        velocity_array = np.diff(distance_array, axis=0)
        velocity_array = np.insert(velocity_array, 0, velocity_array[0], axis=0)

        for i, part in enumerate(parts):
            col_name = f'roi_velocity_{part}_animal_{animal_index}'
            self.df[col_name] = velocity_array[:, i]

        return velocity_array

    def ROI_head_angle(self, animal_index: int, loom_index: int, **kwargs) -> float:
        """Calculate average head angle to loom center over a window."""
        if self.roi_data['loom_circle'] is None:
            return np.nan

        angle_col = f'angle_to_loom_{animal_index}'
        if angle_col not in self.df.columns:
            loom_x, loom_y = self.roi_data['loom_circle']['center']

            nose_x = self.df[f"nose_{animal_index}_x"].to_numpy()
            nose_y = self.df[f"nose_{animal_index}_y"].to_numpy()
            head_x = self.df[f"head_{animal_index}_x"].to_numpy()
            head_y = self.df[f"head_{animal_index}_y"].to_numpy()

            v1x = nose_x - head_x
            v1y = nose_y - head_y
            v2x = loom_x - head_x
            v2y = loom_y - head_y

            angles = np.degrees(np.arctan2(v2y, v2x) - np.arctan2(v1y, v1x))
            normalized_angles = np.where(angles > 180, 360 - angles, angles)

            self.df[angle_col] = normalized_angles

        start_frame, end_frame = self.window_cropper(animal_index, loom_index, **kwargs["window"])
        return self.df[angle_col][start_frame:end_frame].mean()

    def ROI_head_facing(self, animal_index: int, loom_index: int) -> float:
        """Calculate percentage of time facing loom."""
        loomtime = self.loom_session_and_end[loom_index]
        facing_col = self._roi_facing_col('Loom', animal_index)

        if facing_col not in self.df.columns:
            return np.nan

        data = self.df[facing_col][loomtime:loomtime + int(self.fps * 7.5)]
        return data.mean() * 100

    # =========================================================================
    # VELOCITY FEATURES
    # =========================================================================

    def MaxVelocityAfterLoom(self, animal_index: int, loom_index: int, **kwargs) -> float:
        """Calculate maximum velocity after loom."""
        features = [
            f'movement_mouse_{animal_index}_nose',
            f'movement_mouse_{animal_index}_tail_base',
            f'movement_mouse_{animal_index}_ear_left',
            f'movement_mouse_{animal_index}_ear_right'
        ]

        velocity_matrix = self.df[features].to_numpy()
        start_frame, _ = self.window_cropper(animal_index, loom_index, **kwargs["window"])

        end_time = int(self.LatencyToNest(animal_index, loom_index))
        if end_time < 0:
            end_time = len(velocity_matrix) - start_frame - 4
        if end_time > self.fps * 10:
            end_time = int(self.fps * 10)

        cropped = velocity_matrix[start_frame:end_time + start_frame + 4]

        try:
            avg_velocity = np.average(cropped, axis=1)
            smooth = np.convolve(avg_velocity, [0.2] * 5, 'valid')
            return float(np.max(smooth))
        except:
            return -1

    def MaxVelocityToNestAfterLoom(self, animal_index: int, loom_index: int, **kwargs) -> float:
        """Calculate maximum velocity towards nest after loom."""
        self.ROIPolygonVelocity(animal_index)
        parts = ['nose', 'ear_left', 'ear_right', 'head', 'body_center', 'tail_base']
        cols = [f'roi_velocity_{part}_animal_{animal_index}' for part in parts]

        velocity_matrix = self.df[cols].to_numpy()
        start_frame, end_time = self.window_cropper(animal_index, loom_index, **kwargs["window"])

        if end_time < 0:
            end_time = len(velocity_matrix) - start_frame - 2
        if end_time > self.fps * 10:
            end_time = int(self.fps * 10)

        cropped = velocity_matrix[start_frame:int(end_time) + start_frame + 2]

        if cropped.size == 0:
            return np.nan

        avg_velocity = np.average(cropped, axis=1)

        if len(avg_velocity) < 3:
            return float(np.max(avg_velocity)) if len(avg_velocity) > 0 else np.nan

        smooth = np.convolve(avg_velocity, [1/3] * 3, 'valid')

        return float(np.max(smooth)) if len(smooth) > 0 else np.nan

    # =========================================================================
    # GENERIC FEATURE AVERAGING
    # =========================================================================

    def feature_post_loom_average(self, animal_index: int, loom_index: int, feature: str, **kwargs) -> float:
        """Calculate average of a feature over a window."""
        crop_to_both = kwargs.get("crop_to_both", False)

        if crop_to_both and self.pair:
            start_frame, end_frame = self.window_cropper_both(loom_index=loom_index, **kwargs["window"])
        else:
            start_frame, end_frame = self.window_cropper(animal_index, loom_index, **kwargs["window"])

        if start_frame == 0 and end_frame == 0:
            return np.nan

        return self.df[feature][start_frame:end_frame].mean()

    # =========================================================================
    # MISSING DATA DETECTION
    # =========================================================================

    def missing_value_estimate(self, loom_index: int, animal_index: int, body_part: str, **kwargs) -> pd.Series:
        """Detect interpolated regions based on constant acceleration."""
        start_frame, end_frame = self.window_cropper(animal_index, loom_index, **kwargs["window"])

        accel_col = f'acceleration_{body_part}_{animal_index}'
        if accel_col not in self.df.columns:
            data = self.df[f'{body_part}_{animal_index}_x'].to_numpy()
            accel = np.diff(data, n=2)
            self.df[accel_col] = np.concatenate(([accel[0], accel[-1]], accel))

        cropped = self.df[accel_col][start_frame:end_frame]
        interpolated = (cropped > -self.interpolation_tolerance) & (cropped < self.interpolation_tolerance)

        bad_points = pd.Series(0, index=np.arange(end_frame - start_frame))
        bout_start = -1
        bout_length = 0

        for i in range(len(interpolated) - 1):
            if not interpolated.iloc[i]:
                bout_start = i + 1
                bout_length = 0
            else:
                bout_length += 1
                if not interpolated.iloc[i + 1]:
                    if bout_length >= self.interpolation_limit:
                        bad_points[bout_start:bout_start + bout_length] = 1
                    bout_length = 0

        return bad_points

    def total_missing_parts(self, loom_index: int, animal_index: int, **kwargs) -> pd.Series:
        """Total missing frames across all body parts."""
        parts = ['nose', 'ear_left', 'ear_right', 'head', 'body_center', 'tail_base']
        total = self.missing_value_estimate(loom_index, animal_index, parts[0], **kwargs)
        for part in parts[1:]:
            total = total * self.missing_value_estimate(loom_index, animal_index, part, **kwargs)
        return total

    # =========================================================================
    # ALL-FUNCTION ITERATOR
    # =========================================================================

    def AllFunction(self, func, postprocess=None, **kwargs) -> Dict:
        """Iterate a function over all animals and loom events."""
        results = {i: [] for i in range(len(self.animal_ids))}

        for animal_idx in range(len(self.animal_ids)):
            for loom_idx in range(len(self.loom_times)):
                result = func(animal_index=animal_idx + 1, loom_index=loom_idx, **kwargs)

                if isinstance(result, (tuple, list)) and len(result) >= 2:
                    if postprocess:
                        processed = [postprocess(result[0]), postprocess(result[1])]
                        if len(result) > 2:
                            processed.extend(result[2:])
                        results[animal_idx].append(processed)
                    else:
                        results[animal_idx].append(result)
                else:
                    if postprocess:
                        results[animal_idx].append(postprocess(result))
                    else:
                        results[animal_idx].append(result)

        return results

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def DataBlock(self, animal_index: int, loom_index: int) -> Dict:
        """Collect all feature values for a given animal and loom."""
        values = {}

        for name, spec in self.CalculatedFeatures.items():
            func = getattr(self, spec["func"])
            post = spec.get("post", None)
            multi = spec.get("multi", False)
            kwargs = spec.get("kwargs", {})

            all_results = self.AllFunction(func, postprocess=post, **kwargs)
            result = all_results[animal_index][loom_index]

            if multi:
                for subname, val in zip(spec["subfeatures"], result[:2]):
                    values[subname] = val
            else:
                values[name] = result

        return values

    def DataExport(self, sample_name: str) -> Dict[int, pd.DataFrame]:
        """Export analysis results for all animals."""
        n_looms = len(self.loom_times)
        dfs = {}

        for animal_idx in range(len(self.animal_ids)):
            data = {name: [None] * n_looms for name in self.CalculatedFeatures}

            for name, spec in self.CalculatedFeatures.items():
                if spec.get("multi") and "subfeatures" in spec:
                    del data[name]
                    for sub in spec["subfeatures"]:
                        data[sub] = [None] * n_looms

            for loom_idx in range(n_looms):
                block = self.DataBlock(animal_idx, loom_idx)
                for fname, val in block.items():
                    data[fname][loom_idx] = val

            df_dict = {
                "Sample": [sample_name] + [None] * (n_looms - 1),
                "AnimalID": [self.animal_ids[animal_idx]] + [None] * (n_looms - 1),
                "Loom#": list(range(1, n_looms + 1))
            }
            df_dict.update(data)

            df = pd.DataFrame(df_dict)
            dfs[animal_idx] = df

            filename = f"{sample_name}_Animal{self.animal_ids[animal_idx]}_analysis.csv"
            path = os.path.join(self.output_directory, filename)
            df.to_csv(path, index=False)
            print(f"  Saved: {filename}")

        return dfs


# =============================================================================
# SNAKEMAKE INTEGRATION
# =============================================================================

def extract_animal_ids_from_sample(sample_name: str) -> tuple:
    """Extract animal IDs from sample name."""
    match = re.search(r'(\d{3})\+(\d{3})', sample_name)
    if match:
        return (match.group(1), match.group(2))

    match = re.search(r'TRAP2?(\d+)', sample_name)
    if match:
        return (match.group(1),)

    return ('1',)


def main_snakemake(snakemake):
    """Snakemake entry point."""
    print("=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    # Get inputs
    features_csv = snakemake.input.features_with_roi
    roi_json = snakemake.input.roi_data
    metadata_json = snakemake.input.metadata
    events_csv = snakemake.input.events_csv

    # Get output directory
    output_dir = os.path.dirname(snakemake.output.done_flag)
    os.makedirs(output_dir, exist_ok=True)

    # Get sample info
    sample_name = snakemake.params.sample
    animal_ids = extract_animal_ids_from_sample(sample_name)

    # Get parameters
    pixels_per_cm = snakemake.params.get('pixels_per_cm', 32)

    print(f"  Sample: {sample_name}")
    print(f"  Animals: {animal_ids}")
    print(f"  Experiment type: {'Paired' if len(animal_ids) == 2 else 'Single'}")
    print(f"  Features: {features_csv}")
    print(f"  Events: {events_csv}")

    # Run analysis
    analyzer = AnimalBehaviorAnalyzer(
        features_csv=features_csv,
        roi_json=roi_json,
        metadata_json=metadata_json,
        events_csv=events_csv,
        animal_ids=animal_ids,
        output_directory=output_dir,
        pixels_per_cm=pixels_per_cm
    )

    dfs = analyzer.DataExport(sample_name)

    # Create completion flag file
    with open(snakemake.output.done_flag, 'w') as f:
        f.write(f"Analysis completed for {sample_name}\n")
        f.write(f"Animals: {animal_ids}\n")
        f.write(f"Experiment type: {'Paired' if len(animal_ids) == 2 else 'Single'}\n")
        f.write(f"Events source: {events_csv}\n")
        f.write(f"Output files:\n")
        for animal_idx in dfs:
            f.write(f"  - {sample_name}_Animal{animal_ids[animal_idx]}_analysis.csv\n")

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        snakemake
        main_snakemake(snakemake)
    except NameError:
        print("Usage: Run via Snakemake or import as module")
        print("  snakemake --cores 1 feature_analysis")
        sys.exit(1)
