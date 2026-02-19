#!/usr/bin/env python3
"""
ROI Feature Extraction Script for Animal Tracking Data
Adds ROI-based features (distance, in-zone, facing) to existing feature extraction output.

Processes nest and loom zones from pipeline-generated ROI JSON (excludes arena/floor).

Author: June (Molas Lab)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from numba import jit, prange
from typing import Dict, List, Optional

# ==================== CONFIGURATION ====================

# Body parts to analyze for each animal
BODY_PARTS = ['nose', 'ear_left', 'ear_right', 'head', 'body_center', 'tail_base']


# ==================== HELPER CLASSES ====================

class SimbaTimer:
    """Timer class for keeping track of start and end-times of calls"""

    def __init__(self, start: bool = False):
        if start:
            self.start_timer()

    def start_timer(self):
        self.timer = time.time()

    def stop_timer(self):
        if not hasattr(self, "timer"):
            self.elapsed_time = -1
            self.elapsed_time_str = "-1"
        else:
            self.elapsed_time = round(time.time() - self.timer, 4)
            self.elapsed_time_str = str(self.elapsed_time)


# ==================== ROI FEATURE EXTRACTION FUNCTIONS ====================

@jit(nopython=True)
def framewise_euclidean_distance_roi(location_1: np.ndarray,
                                     location_2: np.ndarray,
                                     px_per_mm: float) -> np.ndarray:
    """
    Find frame-wise distances between a moving location and static ROI center in millimeters.
    """
    results = np.full((location_1.shape[0]), np.nan)
    for i in prange(location_1.shape[0]):
        results[i] = np.linalg.norm(location_1[i] - location_2) / px_per_mm
    return results


@jit(nopython=True)
def framewise_inside_polygon_roi(bp_location: np.ndarray, roi_coords: np.ndarray) -> np.ndarray:
    """
    Frame-wise detection if body part is inside static polygon ROI.
    Uses ray casting algorithm.
    """
    results = np.full((bp_location.shape[0]), 0)
    for i in prange(0, results.shape[0]):
        x, y, n = bp_location[i][0], bp_location[i][1], len(roi_coords)
        p2x, p2y, xints, inside = 0.0, 0.0, 0.0, False
        p1x, p1y = roi_coords[0]
        for j in prange(n + 1):
            p2x, p2y = roi_coords[j % n]
            if (
                (y > min(p1y, p2y))
                and (y <= max(p1y, p2y))
                and (x <= max(p1x, p2x))
            ):
                if p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xints:
                    inside = not inside
            p1x, p1y = p2x, p2y
        if inside:
            results[i] = 1

    return results


@jit(nopython=True)
def framewise_inside_circle_roi(bp_location: np.ndarray,
                                 center: np.ndarray,
                                 radius: float) -> np.ndarray:
    """
    Frame-wise detection if body part is inside a circular ROI.
    """
    results = np.full((bp_location.shape[0]), 0)
    for i in prange(bp_location.shape[0]):
        dist = np.sqrt((bp_location[i][0] - center[0])**2 +
                       (bp_location[i][1] - center[1])**2)
        if dist <= radius:
            results[i] = 1
    return results


@jit(nopython=True)
def calculate_facing_direction(nose_coords: np.ndarray,
                               body_center_coords: np.ndarray,
                               roi_center: np.ndarray,
                               angle_threshold: float = 45.0) -> np.ndarray:
    """
    Calculate if animal is facing towards ROI based on nose and body center positions.
    """
    results = np.full((nose_coords.shape[0]), 0)

    for i in prange(nose_coords.shape[0]):
        heading_vec = nose_coords[i] - body_center_coords[i]
        to_roi_vec = roi_center - nose_coords[i]

        heading_mag = np.linalg.norm(heading_vec)
        to_roi_mag = np.linalg.norm(to_roi_vec)

        if heading_mag > 0 and to_roi_mag > 0:
            cos_angle = np.dot(heading_vec, to_roi_vec) / (heading_mag * to_roi_mag)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_deg = np.degrees(np.arccos(cos_angle))

            if angle_deg <= angle_threshold:
                results[i] = 1

    return results


# ==================== ROI DATA LOADING ====================

def load_roi_from_json(roi_json_path: str) -> dict:
    """
    Load ROI definitions from pipeline-generated JSON file.

    New format contains 'regions' with floor, nest, and loom zones.
    Only nest and loom are processed (arena/floor is excluded from analysis).

    :param roi_json_path: Path to ROI JSON file
    :return: Dictionary with ROI definitions (polygons for nest, circles for loom)
    """
    rois = {
        'polygons': [],
        'circles': []
    }

    if not os.path.exists(roi_json_path):
        print(f"  Warning: ROI file not found: {roi_json_path}")
        return rois

    with open(roi_json_path, 'r') as f:
        roi_data = json.load(f)

    # New format with regions
    if 'regions' in roi_data:
        for region in roi_data['regions']:
            # Extract nest polygon (exclude floor/arena)
            if 'nest' in region:
                nest_corners = np.array(region['nest'], dtype=np.float32)
                nest_center = nest_corners.mean(axis=0)
                rois['polygons'].append({
                    'name': 'Nest',
                    'center': nest_center,
                    'vertices': nest_corners
                })
                print(f"  Loaded Nest polygon with {len(nest_corners)} vertices")

            # Extract loom circle
            if 'loom' in region:
                loom_data = region['loom']
                loom_center = np.array(loom_data[0], dtype=np.float32)
                loom_radius = float(loom_data[1])
                rois['circles'].append({
                    'name': 'Loom',
                    'center': loom_center,
                    'radius': loom_radius
                })
                print(f"  Loaded Loom circle at {loom_center} with radius {loom_radius}")

    # Legacy format with corners (only arena - skip for this analysis)
    elif 'corners' in roi_data and roi_data.get('confidence', 0) > 0:
        print("  Warning: Legacy ROI format detected - no nest/loom zones available")
        # Don't add arena to the analysis - it's excluded per requirements

    return rois


# ==================== MAIN ROI FEATURE EXTRACTOR ====================

class ROIFeatureExtractor:
    """Extract ROI-based features from pose estimation data"""

    def __init__(self, px_per_mm: float = 1.47683):
        self.px_per_mm = px_per_mm
        self.body_parts = BODY_PARTS

    def detect_animals_and_bodyparts(self, df: pd.DataFrame) -> Dict:
        """
        Detect which animals and body parts are present in the data.
        """
        animal_bodyparts = {}

        for col in df.columns:
            if not col.endswith('_x'):
                continue

            parts = col[:-2].split('_')

            if len(parts) >= 2 and parts[-1].isdigit():
                animal_id = parts[-1]
                bodypart = '_'.join(parts[:-1])

                if bodypart in self.body_parts:
                    if animal_id not in animal_bodyparts:
                        animal_bodyparts[animal_id] = set()
                    animal_bodyparts[animal_id].add(bodypart)

        return animal_bodyparts

    def extract_roi_features(self, data_df: pd.DataFrame, rois: dict,
                           animal_bodyparts: dict) -> pd.DataFrame:
        """
        Extract ROI features from pose data.
        Processes both polygon (nest) and circle (loom) ROIs.

        Column naming follows SimBA convention:
            - Distance: "{roi_name} {animal_name} {bodypart}_{animal_id} distance"
            - In-zone:  "{roi_name} {animal_name} {bodypart}_{animal_id} in zone"
            - Facing:   "{roi_name} {animal_name} facing"

        Examples:
            - "nest Animal_1 nose_1 in zone"
            - "Loom Animal_2 body_center_2 distance"
            - "nest Animal_1 facing"
        """
        out_df = pd.DataFrame()

        if len(rois['circles']) == 0 and len(rois['polygons']) == 0:
            print("  No ROIs to process (nest/loom zones not found)")
            return out_df

        print(f"  Processing ROIs for {len(animal_bodyparts)} animal(s)...")
        print(f"    - Polygons (Nest): {len(rois['polygons'])}")
        print(f"    - Circles (Loom): {len(rois['circles'])}")

        for animal_id, bodyparts in sorted(animal_bodyparts.items()):
            animal_name = f"Animal_{animal_id}"
            print(f"    - {animal_name}: {len(bodyparts)} body parts")

            for bodypart in sorted(bodyparts):
                bodypart_col_name = f"{bodypart}_{animal_id}"

                x_col = f"{bodypart_col_name}_x"
                y_col = f"{bodypart_col_name}_y"

                if x_col not in data_df.columns or y_col not in data_df.columns:
                    continue

                bp_location = data_df[[x_col, y_col]].values.astype(np.float32)

                # Process polygon ROIs (Nest)
                for roi in rois['polygons']:
                    roi_name = roi['name']
                    roi_center = roi['center']
                    roi_vertices = roi['vertices']

                    # Distance to center (SimBA format: "nest Animal_1 nose_1 distance")
                    dist_col = f"{roi_name} {animal_name} {bodypart_col_name} distance"
                    out_df[dist_col] = framewise_euclidean_distance_roi(
                        bp_location, roi_center, self.px_per_mm
                    )

                    # In-zone detection (SimBA format: "nest Animal_1 nose_1 in zone")
                    zone_col = f"{roi_name} {animal_name} {bodypart_col_name} in zone"
                    out_df[zone_col] = framewise_inside_polygon_roi(
                        bp_location, roi_vertices
                    )

                # Process circle ROIs (Loom)
                for roi in rois['circles']:
                    roi_name = roi['name']  # Keep capitalization for Loom
                    roi_center = roi['center']
                    roi_radius = roi['radius']

                    # Distance to center (SimBA format: "Loom Animal_1 nose_1 distance")
                    dist_col = f"{roi_name} {animal_name} {bodypart_col_name} distance"
                    out_df[dist_col] = framewise_euclidean_distance_roi(
                        bp_location, roi_center, self.px_per_mm
                    )

                    # In-zone detection (SimBA format: "Loom Animal_1 nose_1 in zone")
                    zone_col = f"{roi_name} {animal_name} {bodypart_col_name} in zone"
                    out_df[zone_col] = framewise_inside_circle_roi(
                        bp_location, roi_center, roi_radius
                    )

            # Calculate "facing" features for both polygon and circle ROIs
            if 'nose' in bodyparts and 'body_center' in bodyparts:
                nose_x = f"nose_{animal_id}_x"
                nose_y = f"nose_{animal_id}_y"
                bc_x = f"body_center_{animal_id}_x"
                bc_y = f"body_center_{animal_id}_y"

                if all(col in data_df.columns for col in [nose_x, nose_y, bc_x, bc_y]):
                    nose_coords = data_df[[nose_x, nose_y]].values.astype(np.float32)
                    bc_coords = data_df[[bc_x, bc_y]].values.astype(np.float32)

                    # Facing polygon ROIs (Nest) - SimBA format: "nest Animal_1 facing"
                    for roi in rois['polygons']:
                        roi_name = roi['name']
                        facing_col = f"{roi_name} {animal_name} facing"
                        out_df[facing_col] = calculate_facing_direction(
                            nose_coords, bc_coords, roi['center']
                        )

                    # Facing circle ROIs (Loom) - SimBA format: "Loom Animal_1 facing"
                    for roi in rois['circles']:
                        roi_name = roi['name']
                        facing_col = f"{roi_name} {animal_name} facing"
                        out_df[facing_col] = calculate_facing_direction(
                            nose_coords, bc_coords, roi['center']
                        )

        return out_df


# ==================== SNAKEMAKE INTEGRATION ====================

def main_snakemake(snakemake):
    """Main entry point for Snakemake integration"""

    input_csv = snakemake.input.features_csv
    roi_json = snakemake.input.roi_data
    metadata_json = snakemake.input.metadata
    output_csv = snakemake.output.features_with_roi

    # Load metadata for px_per_mm
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)

    # Use default px_per_mm from config or metadata
    px_per_mm = snakemake.params.get('px_per_mm', 1.47683)

    print("=" * 70)
    print("ROI FEATURE EXTRACTION (Snakemake Mode)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Pixels per mm: {px_per_mm}")
    print(f"  Input: {input_csv}")
    print(f"  ROI: {roi_json}")
    print(f"  Output: {output_csv}")
    print("=" * 70)

    timer = SimbaTimer(start=True)

    # Read input data
    print("\nLoading input data...")
    data_df = pd.read_csv(input_csv, index_col=0)
    print(f"  Loaded {len(data_df)} frames with {len(data_df.columns)} features")

    # Load ROI definitions (nest and loom only - arena excluded)
    print("\nLoading ROI definitions (nest and loom zones)...")
    rois = load_roi_from_json(roi_json)
    print(f"  Found {len(rois['polygons'])} polygon(s) - Nest zones")
    print(f"  Found {len(rois['circles'])} circle(s) - Loom zones")

    # Initialize extractor
    extractor = ROIFeatureExtractor(px_per_mm=px_per_mm)

    # Detect animals and body parts
    print("\nDetecting animals and body parts...")
    animal_bodyparts = extractor.detect_animals_and_bodyparts(data_df)

    if not animal_bodyparts:
        print("  Warning: No animals detected in data!")
        data_df.to_csv(output_csv, index=True)
        return

    # Extract ROI features
    print("\nExtracting ROI features...")
    roi_features = extractor.extract_roi_features(data_df, rois, animal_bodyparts)

    # Combine features
    if len(roi_features.columns) > 0:
        print("\nCombining features...")
        output_df = pd.concat([data_df, roi_features], axis=1)
    else:
        print("\nNo ROI features extracted, using original data...")
        output_df = data_df

    # Save output
    print(f"\nSaving output to: {output_csv}")
    output_df.to_csv(output_csv, index=True)

    timer.stop_timer()

    print("=" * 70)
    print(f"SUCCESS!")
    print(f"  Original features: {len(data_df.columns)}")
    print(f"  ROI features added: {len(roi_features.columns) if len(roi_features.columns) > 0 else 0}")
    print(f"  Total features: {len(output_df.columns)}")
    print(f"  Elapsed time: {timer.elapsed_time_str}s")
    print("=" * 70)


# ==================== MAIN ====================

if __name__ == "__main__":
    # Check if running under Snakemake
    try:
        snakemake
        main_snakemake(snakemake)
    except NameError:
        # CLI mode
        if len(sys.argv) != 4:
            print("Usage: python roi_feature_extraction.py <input_csv> <roi_json> <output_csv>")
            sys.exit(1)

        # Simple CLI implementation
        input_csv = sys.argv[1]
        roi_json = sys.argv[2]
        output_csv = sys.argv[3]

        data_df = pd.read_csv(input_csv, index_col=0)
        rois = load_roi_from_json(roi_json)
        extractor = ROIFeatureExtractor()
        animal_bodyparts = extractor.detect_animals_and_bodyparts(data_df)
        roi_features = extractor.extract_roi_features(data_df, rois, animal_bodyparts)

        if len(roi_features.columns) > 0:
            output_df = pd.concat([data_df, roi_features], axis=1)
        else:
            output_df = data_df

        output_df.to_csv(output_csv, index=True)
        print(f"Saved to {output_csv}")
