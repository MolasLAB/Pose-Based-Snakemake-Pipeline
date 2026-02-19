#!/usr/bin/env python3
"""
Standalone Feature Extraction Script for Two-Animal Social Interaction Analysis
Extracts behavioral features from SLEAP pose estimation data.

Usage:
    python standalone_feature_extraction.py <input_csv> <output_csv>
    
Example:
    python standalone_feature_extraction.py input_data.csv output_features.csv
"""

import os
import sys
import math
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from numba import jit, njit, prange
from numba.np.extensions import cross2d
from numpy import arccos as np_arccos, clip as np_clip, dot as np_dot
from numpy.linalg import norm as linalg_norm
from math import degrees as mathdegrees


# ==================== CONFIGURATION ====================

PX_PER_MM = 1.47683
FPS = 29.595
ROLL_WINDOWS_VALUES = [3, 6, 9, 15]



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


# ==================== PERIMETER/HULL FUNCTIONS ====================

@njit("(float32[:,:], int64[:], int64, int64)")
def process(S, P, a, b):
    """Helper function for convex hull calculation"""
    signed_dist = cross2d(S[P] - S[a], S[b] - S[a])
    K = np.array(
        [i for s, i in zip(signed_dist, P) if s > 0 and i != a and i != b],
        dtype=np.int64,
    )
    if len(K) == 0:
        return [a, b]
    c = P[np.argmax(signed_dist)]
    return process(S, K, a, c)[:-1] + process(S, K, c, b)


@njit("(float32[:, :, :], types.unicode_type)", fastmath=True)
def jitted_hull(points: np.ndarray, target: str = "perimeter") -> np.ndarray:
    """
    Compute attributes (e.g., perimeter or area) of a polygon.
    
    :param array points: 3d array FRAMESxBODY-PARTxCOORDINATE
    :param str target: Options [perimeter, area]
    :return: 1d np.array representing perimeter length or area of polygon on each frame
    """
    
    def perimeter(xy):
        perimeter = np.linalg.norm(xy[0] - xy[-1])
        for i in prange(xy.shape[0] - 1):
            p = np.linalg.norm(xy[i] - xy[i + 1])
            perimeter += p
        return perimeter
    
    def area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    results = np.full((points.shape[0]), np.nan)
    for i in range(points.shape[0]):
        S = points[i, :, :]
        a, b = np.argmin(S[:, 0]), np.argmax(S[:, 0])
        max_index = np.argmax(S[:, 0])
        idx = (
            process(S, np.arange(S.shape[0]), a, max_index)[:-1]
            + process(S, np.arange(S.shape[0]), max_index, a)[:-1]
        )
        x, y = np.full((len(idx)), np.nan), np.full((len(idx)), np.nan)
        for j in prange(len(idx)):
            x[j], y[j] = S[idx[j], 0], S[idx[j], 1]
        x0, y0 = np.mean(x), np.mean(y)
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        angles = np.where(
            (y - y0) > 0, np.arccos((x - x0) / r), 2 * np.pi - np.arccos((x - x0) / r)
        )
        mask = np.argsort(angles)
        x_sorted, y_sorted = x[mask], y[mask]
        if target == "perimeter":
            xy = np.vstack((x_sorted, y_sorted)).T
            results[i] = perimeter(xy)
        if target == "area":
            results[i] = area(x_sorted, y_sorted)
    
    return results


# ==================== FEATURE EXTRACTION FUNCTIONS ====================
# Orignally taken from SIMBA scripts


@jit(nopython=True)
def euclidean_distance(bp_1_x: np.ndarray,
                       bp_2_x: np.ndarray,
                       bp_1_y: np.ndarray,
                       bp_2_y: np.ndarray,
                       px_per_mm: float) -> np.ndarray:
    """
    Compute Euclidean distance in millimeters between two body-parts.
    """
    series = (np.sqrt((bp_1_x - bp_2_x) ** 2 + (bp_1_y - bp_2_y) ** 2)) / px_per_mm
    return series


@jit(nopython=True, fastmath=True)
def angle3pt_vectorized(data: np.ndarray) -> np.ndarray:
    """
    Numba accelerated compute of frame-wise 3-point angles.
    
    :param ndarray data: 2D numerical array with frame number on x and [ax, ay, bx, by, cx, cy] on y.
    :return: 1d float numerical array of size data.shape[0] with angles.
    """
    results = np.full((data.shape[0]), 0.0)
    for i in prange(data.shape[0]):
        angle = math.degrees(
            math.atan2(data[i][5] - data[i][3], data[i][4] - data[i][2])
            - math.atan2(data[i][1] - data[i][3], data[i][0] - data[i][2])
        )
        if angle < 0:
            angle += 360
        results[i] = angle
    
    return results


def angle4pt_serialized(data: np.ndarray):
    """
    Calculate angles between two lines defined by 4 points.
    Takes 4 points: first two form line 1, last two form line 2.
    Returns angle between the lines.
    """
    resultsA = np.full((data.shape[0]), 0.0)
    for i in prange(data.shape[0]):
        vector1 = [data[i][2] - data[i][0], data[i][3] - data[i][1]]
        vector2 = [data[i][6] - data[i][4], data[i][7] - data[i][5]]
        unitvector1 = vector1 / linalg_norm(vector1)
        unitvector2 = vector2 / linalg_norm(vector2)
        
        resultsA[i] = mathdegrees(np_arccos(np_clip(np_dot(unitvector1, unitvector2), -1.0, 1.0)))
    return resultsA


@jit(nopython=True)
def cdist(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """
    Analogue of scipy.cdist for two 2D arrays.
    Calculate Euclidean distances between all coordinates in one array 
    and all coordinates in a second array.
    """
    results = np.full((array_1.shape[0], array_2.shape[0]), np.nan)
    for i in prange(array_1.shape[0]):
        for j in prange(array_2.shape[0]):
            results[i][j] = np.linalg.norm(array_1[i] - array_2[j])
    return results


@jit(nopython=True)
def count_values_in_range(data: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """
    Jitted helper finding count of values that falls within ranges.
    E.g., count number of pose-estimated body-parts that fall within 
    defined bracket of probabilities per frame.
    
    :param np.ndarray data: 2D numpy array with frames on X.
    :param np.ndarray ranges: 2D numpy array representing the brackets. E.g., [[0, 0.1], [0.1, 0.5]]
    :return: 2D numpy array of size data.shape[0], ranges.shape[1]
    """
    results = np.full((data.shape[0], ranges.shape[0]), 0)
    for i in prange(data.shape[0]):
        for j in prange(ranges.shape[0]):
            lower_bound, upper_bound = ranges[j][0], ranges[j][1]
            results[i][j] = data[i][
                np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)
            ].shape[0]
    return results


# ==================== DATA LOADING AND RESHAPING ====================

def get_fn_ext(filepath: str):
    """Split file path into directory, filename, and extension."""
    file_extension = Path(filepath).suffix
    file_name = os.path.basename(filepath.rsplit(file_extension, 1)[0])
    file_dir = os.path.dirname(filepath)
    return file_dir, file_name, file_extension


def read_sleap_csv(file_path: str) -> pd.DataFrame:
    """
    Read SLEAP tracking CSV and reshape into expected format.
    
    Expected SLEAP format has columns:
    - track, frame_idx, instance.score, nose.x, nose.y, nose.score, ...
    
    Reshapes to format with columns:
    - nose_1_x, nose_1_y, nose_1_p, ear_left_1_x, ...
    """
    print(f"Reading input file: {file_path}")
    df = pd.read_csv(file_path)
    
    # Group by frame_idx and track
    grouped = df.groupby(['frame_idx', 'track'])
    
    # Initialize output dataframe
    max_frame = df['frame_idx'].max()
    
    # Body part names (update based on your data)
    body_parts = ['nose', 'ear_left', 'ear_right', 'head', 'body_center', 'tail_base']
    
    # Check if we have two animals
    unique_tracks = sorted(df['track'].unique())
    num_animals = len(unique_tracks)
    
    if num_animals == 2:
        print("Detected 2 animals (paired analysis)")
        paired = True
        # Create column names for both animals
        columns = []
        for animal_id in [1, 2]:
            for bp in body_parts:
                columns.extend([f'{bp}_{animal_id}_x', f'{bp}_{animal_id}_y', f'{bp}_{animal_id}_p'])
    else:
        print("Detected 1 animal (single analysis)")
        paired = False
        # Create column names for single animal
        columns = []
        for bp in body_parts:
            columns.extend([f'{bp}_1_x', f'{bp}_1_y', f'{bp}_1_p'])
    
    # Initialize output array
    output_data = np.zeros((max_frame + 1, len(columns)))
    
    # Fill in data - create track-to-animal mapping outside the loop
    # Map tracks to animals: track 0 -> animal 1, track 1 -> animal 2
    if paired:
        track_to_animal = {unique_tracks[0]: 1, unique_tracks[1]: 2}
        print(f"  Track mapping: {unique_tracks[0]} -> Animal 1, {unique_tracks[1]} -> Animal 2")
    else:
        track_to_animal = {unique_tracks[0]: 1}
        print(f"  Track mapping: {unique_tracks[0]} -> Animal 1")
    
    for (frame_idx, track), group in grouped:
        if track >= num_animals:
            continue
        
        # Get animal_id from the track mapping
        animal_id = track_to_animal[track]
        
        for bp_idx, bp in enumerate(body_parts):
            x_col = f'{bp}.x'
            y_col = f'{bp}.y'
            score_col = f'{bp}.score'
            
            if x_col in group.columns:
                x_val = group[x_col].values[0]
                y_val = group[y_col].values[0]
                score_val = group[score_col].values[0]
                
                # Find column indices in output
                if paired:
                    base_idx = (animal_id - 1) * len(body_parts) * 3 + bp_idx * 3
                else:
                    base_idx = bp_idx * 3
                
                output_data[frame_idx, base_idx] = x_val
                output_data[frame_idx, base_idx + 1] = y_val
                output_data[frame_idx, base_idx + 2] = score_val
    
    # Create DataFrame
    output_df = pd.DataFrame(output_data, columns=columns)
    
    print(f"Reshaped data: {len(output_df)} frames, {len(columns)} columns")
    
    return output_df, paired


# ==================== MAIN FEATURE EXTRACTION CLASS ====================

class FeatureExtractor:
    """Feature extractor for two-animal social behavior analysis"""
    
    def __init__(self, px_per_mm: float = PX_PER_MM, fps: float = FPS):
        self.px_per_mm = px_per_mm
        self.fps = fps
        self.roll_windows_values = ROLL_WINDOWS_VALUES
        self.angle3pt_serialized = angle3pt_vectorized
        self.euclidean_distance = euclidean_distance
        self.cdist = cdist
        self.count_values_in_range = count_values_in_range
    
    def extract_features(self, in_data: pd.DataFrame, paired: bool):
        """
        Main feature extraction method.
        
        :param in_data: Input DataFrame with pose data
        :param paired: Whether this is paired (2 animals) or single animal
        :return: DataFrame with extracted features
        """
        timer = SimbaTimer(start=True)
        
        # Set up headers
        if paired:
            self.in_headers = ['nose_1_x', 'nose_1_y', 'nose_1_p', 'ear_left_1_x', 'ear_left_1_y', 
                              'ear_left_1_p', 'ear_right_1_x', 'ear_right_1_y', 'ear_right_1_p', 
                              'head_1_x', 'head_1_y', 'head_1_p', 'body_center_1_x', 'body_center_1_y', 
                              'body_center_1_p', 'tail_base_1_x', 'tail_base_1_y', 'tail_base_1_p',
                              'nose_2_x', 'nose_2_y', 'nose_2_p', 'head_2_x', 'head_2_y', 'head_2_p',
                              'ear_left_2_x', 'ear_left_2_y', 'ear_left_2_p', 'ear_right_2_x', 
                              'ear_right_2_y', 'ear_right_2_p', 'body_center_2_x', 'body_center_2_y', 
                              'body_center_2_p', 'tail_base_2_x', 'tail_base_2_y', 'tail_base_2_p']
            self.mouse_1_headers, self.mouse_2_headers = (self.in_headers[0:18], self.in_headers[18:],)
            self.mouse_2_p_headers = [x for x in self.mouse_2_headers if x[-2:] == "_p"]
            self.mouse_1_p_headers = [x for x in self.mouse_1_headers if x[-2:] == "_p"]
            self.mouse_1_headers = [x for x in self.mouse_1_headers if x[-2:] != "_p"]
            self.mouse_2_headers = [x for x in self.mouse_2_headers if x[-2:] != "_p"]
        else:
            self.in_headers = ['nose_1_x', 'nose_1_y', 'nose_1_p', 'ear_left_1_x', 'ear_left_1_y', 
                              'ear_left_1_p', 'ear_right_1_x', 'ear_right_1_y', 'ear_right_1_p', 
                              'head_1_x', 'head_1_y', 'head_1_p', 'body_center_1_x', 'body_center_1_y', 
                              'body_center_1_p', 'tail_base_1_x', 'tail_base_1_y', 'tail_base_1_p']
            self.mouse_1_headers = self.in_headers[0:18]
            self.mouse_1_p_headers = [x for x in self.mouse_1_headers if x[-2:] == "_p"]
            self.mouse_1_headers = [x for x in self.mouse_1_headers if x[-2:] != "_p"]
        
        self.paired = paired
        
        # Fill NaN values and convert to numeric
        in_data = in_data.fillna(0).apply(pd.to_numeric).reset_index(drop=True)
        
        print(f"Processing {len(in_data)} frames...")
        
        # Start with input data
        self.out_data = deepcopy(in_data)
        
        # Calculate poly area (convex hull perimeter)
        print("Calculating convex hull areas...")
        mouse_1_ar = np.reshape(
            self.out_data[self.mouse_1_headers].values,
            (len(self.out_data), -1, 2),
        ).astype(np.float32)
        self.out_data["poly_area_1"] = (
            jitted_hull(points=mouse_1_ar, target="perimeter")
            / self.px_per_mm
        )
        
        if self.paired:
            mouse_2_ar = np.reshape(
                self.out_data[self.mouse_2_headers].values,
                (len(self.out_data), -1, 2),
            ).astype(np.float32)
            self.out_data["poly_area_2"] = (
                jitted_hull(points=mouse_2_ar, target="perimeter")
                / self.px_per_mm
            )
        
        # Calculate distance features
        print("Calculating distance features...")
        self.out_data["nose_to_tail_1"] = self.euclidean_distance(
            self.out_data["nose_1_x"].values,
            self.out_data["tail_base_1_x"].values,
            self.out_data["nose_1_y"].values,
            self.out_data["tail_base_1_y"].values,
            self.px_per_mm,
        )
        
        self.out_data["ear_distance_1"] = self.euclidean_distance(
            self.out_data["ear_left_1_x"].values,
            self.out_data["ear_right_1_x"].values,
            self.out_data["ear_left_1_y"].values,
            self.out_data["ear_right_1_y"].values,
            self.px_per_mm,
        )
        
        if self.paired:
            self.out_data["nose_to_tail_2"] = self.euclidean_distance(
                self.out_data["nose_2_x"].values,
                self.out_data["tail_base_2_x"].values,
                self.out_data["nose_2_y"].values,
                self.out_data["tail_base_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["ear_distance_2"] = self.euclidean_distance(
                self.out_data["ear_left_2_x"].values,
                self.out_data["ear_right_2_x"].values,
                self.out_data["ear_left_2_y"].values,
                self.out_data["ear_right_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["nose_to_nose_distance"] = self.euclidean_distance(
                self.out_data["nose_1_x"].values,
                self.out_data["nose_2_x"].values,
                self.out_data["nose_1_y"].values,
                self.out_data["nose_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["body_to_body_distance"] = self.euclidean_distance(
                self.out_data["body_center_1_x"].values,
                self.out_data["body_center_2_x"].values,
                self.out_data["body_center_1_y"].values,
                self.out_data["body_center_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["tail_to_tail_distance"] = self.euclidean_distance(
                self.out_data["tail_base_1_x"].values,
                self.out_data["tail_base_2_x"].values,
                self.out_data["tail_base_1_y"].values,
                self.out_data["tail_base_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["nose_1_to_tail_2_distance"] = self.euclidean_distance(
                self.out_data["nose_1_x"].values,
                self.out_data["tail_base_2_x"].values,
                self.out_data["nose_1_y"].values,
                self.out_data["tail_base_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["nose_2_to_tail_1_distance"] = self.euclidean_distance(
                self.out_data["nose_2_x"].values,
                self.out_data["tail_base_1_x"].values,
                self.out_data["nose_2_y"].values,
                self.out_data["tail_base_1_y"].values,
                self.px_per_mm,
            )
        
        # Calculate angle features
        print("Calculating angle features...")
        self.out_data["body_center_angle_1"] = self.angle3pt_serialized(np.column_stack([
            self.out_data["head_1_x"].values,
            self.out_data["head_1_y"].values,
            self.out_data["body_center_1_x"].values,
            self.out_data["body_center_1_y"].values,
            self.out_data["tail_base_1_x"].values,
            self.out_data["tail_base_1_y"].values]))
        
        if self.paired:
            self.out_data["body_center_angle_2"] = self.angle3pt_serialized(np.column_stack([
                self.out_data["head_2_x"].values,
                self.out_data["head_2_y"].values,
                self.out_data["body_center_2_x"].values,
                self.out_data["body_center_2_y"].values,
                self.out_data["tail_base_2_x"].values,
                self.out_data["tail_base_2_y"].values]))
        
        # Nose angles (continuous, mapped to -180 to 180)
        tempangle1 = self.angle3pt_serialized(np.column_stack([
            self.out_data["ear_left_1_x"].values,
            self.out_data["ear_left_1_y"].values,
            self.out_data["nose_1_x"].values,
            self.out_data["nose_1_y"].values,
            self.out_data["ear_right_1_x"].values,
            self.out_data["ear_right_1_y"].values]))
        
        for i in range(len(tempangle1)):
            if tempangle1[i] > 180:
                tempangle1[i] -= 360
        
        self.out_data["nose_angle_1"] = tempangle1
        
        if self.paired:
            tempangle2 = self.angle3pt_serialized(np.column_stack([
                self.out_data["ear_left_2_x"].values,
                self.out_data["ear_left_2_y"].values,
                self.out_data["nose_2_x"].values,
                self.out_data["nose_2_y"].values,
                self.out_data["ear_right_2_x"].values,
                self.out_data["ear_right_2_y"].values]))
            
            for i in range(len(tempangle2)):
                if tempangle2[i] > 180:
                    tempangle2[i] -= 360
            
            self.out_data["nose_angle_2"] = tempangle2
            
            # Relative angles between mice
            self.out_data["relative_midline_angle"] = angle4pt_serialized(np.column_stack([
                self.out_data["nose_1_x"].values,
                self.out_data["nose_1_y"].values,
                self.out_data["tail_base_1_x"].values,
                self.out_data["tail_base_1_y"].values,
                self.out_data["nose_2_x"].values,
                self.out_data["nose_2_y"].values,
                self.out_data["tail_base_2_x"].values,
                self.out_data["tail_base_2_y"].values]))
            
            self.out_data["relative_nose_head_angle"] = angle4pt_serialized(np.column_stack([
                self.out_data["nose_1_x"].values,
                self.out_data["nose_1_y"].values,
                self.out_data["head_1_x"].values,
                self.out_data["head_1_y"].values,
                self.out_data["nose_2_x"].values,
                self.out_data["nose_2_y"].values,
                self.out_data["head_2_x"].values,
                self.out_data["head_2_y"].values]))
        
        # Calculate movement features
        print("Calculating movement features...")
        self.in_data_shifted = (
            self.out_data.shift(periods=1).add_suffix("_shifted").bfill()
        )
        self.in_data = (
            pd.concat([in_data, self.in_data_shifted], axis=1, join="inner")
            .fillna(0)
            .reset_index(drop=True)
        )
        
        # Movement features setup
        individual_movement_features_x_y = ["nose", "head", "ear_left", "ear_right", "body_center", "tail_base"]
        individual_movement_features_1d = ["poly_area", "nose_to_tail", "ear_distance", "body_center_angle", "nose_angle"]
        
        if self.paired:
            paired_movement_features = ["nose_to_nose_distance", "tail_to_tail_distance", "relative_midline_angle", "relative_nose_head_angle"]
            animal_indices = [1, 2]
        else:
            paired_movement_features = []
            animal_indices = [1]
        
        for animal_number in animal_indices:
            for feature_xy in individual_movement_features_x_y:
                self.out_data[f"movement_mouse_{animal_number}_{feature_xy}"] = self.euclidean_distance(
                    self.in_data[f"{feature_xy}_{animal_number}_x_shifted"].values,
                    self.in_data[f"{feature_xy}_{animal_number}_x"].values,
                    self.in_data[f"{feature_xy}_{animal_number}_y_shifted"].values,
                    self.in_data[f"{feature_xy}_{animal_number}_y"].values,
                    self.px_per_mm,
                )
            
            for feature_s in individual_movement_features_1d:
                self.out_data[f"movement_mouse_{animal_number}_{feature_s}"] = (
                    self.in_data[f"{feature_s}_{animal_number}_shifted"]
                    - self.out_data[f"{feature_s}_{animal_number}"]
                )
        
        for feature_p in paired_movement_features:
            self.out_data[f"movement_{feature_p}"] = (
                self.in_data[f"{feature_p}_shifted"]
                - self.out_data[f"{feature_p}"]
            )
        
        # Calculate hull variables
        print("Calculating hull variables...")
        self.hull_dict = defaultdict(list)
        
        if self.paired:
            mouse_1_array, mouse_2_array = (
                self.in_data[self.mouse_1_headers].to_numpy(),
                self.in_data[self.mouse_2_headers].to_numpy(),
            )
            for cnt, (animal_1, animal_2) in enumerate(zip(mouse_1_array, mouse_2_array)):
                animal_1, animal_2 = np.reshape(animal_1, (-1, 2)), np.reshape(animal_2, (-1, 2))
                animal_1_dist, animal_2_dist = self.cdist(animal_1, animal_1), self.cdist(animal_2, animal_2)
                animal_1_dist, animal_2_dist = (
                    animal_1_dist[animal_1_dist != 0],
                    animal_2_dist[animal_2_dist != 0],
                )
                for animal, animal_name in zip([animal_1_dist, animal_2_dist], ["m1", "m2"]):
                    self.hull_dict[f"{animal_name}_hull_large_euclidean"].append(
                        np.amax(animal, initial=0) / self.px_per_mm)
                    self.hull_dict[f"{animal_name}_hull_small_euclidean"].append(
                        np.min(animal, initial=self.hull_dict[f"{animal_name}_hull_large_euclidean"][-1])
                        / self.px_per_mm
                    )
                    self.hull_dict[f"{animal_name}_hull_mean_euclidean"].append(
                        np.mean(animal) / self.px_per_mm
                    )
                    self.hull_dict[f"{animal_name}_hull_sum_euclidean"].append(
                        np.sum(animal, initial=0) / self.px_per_mm
                    )
            
            for k, v in self.hull_dict.items():
                self.out_data[k] = v
        else:
            mouse_1_array = self.in_data[self.mouse_1_headers].to_numpy()
            for cnt, animal_1 in enumerate(mouse_1_array):
                animal_1 = np.reshape(animal_1, (-1, 2))
                animal_1_dist = self.cdist(animal_1, animal_1)
                animal_1_dist = animal_1_dist[animal_1_dist != 0]
                for animal, animal_name in zip([animal_1_dist], ["m1"]):
                    self.hull_dict[f"{animal_name}_hull_large_euclidean"].append(
                        np.amax(animal, initial=0) / self.px_per_mm)
                    self.hull_dict[f"{animal_name}_hull_small_euclidean"].append(
                        np.min(animal, initial=self.hull_dict[f"{animal_name}_hull_large_euclidean"][-1])
                        / self.px_per_mm
                    )
                    self.hull_dict[f"{animal_name}_hull_mean_euclidean"].append(
                        np.mean(animal) / self.px_per_mm
                    )
                    self.hull_dict[f"{animal_name}_hull_sum_euclidean"].append(
                        np.sum(animal, initial=0) / self.px_per_mm
                    )
            
            for k, v in self.hull_dict.items():
                self.out_data[k] = v
        
        # Total movement features
        self.out_data["total_movement_all_bodyparts_m1"] = self.out_data.eval(
            "movement_mouse_1_nose + movement_mouse_1_tail_base + movement_mouse_1_ear_left + "
            "movement_mouse_1_ear_right + movement_mouse_1_head + movement_mouse_1_body_center"
        )
        
        if self.paired:
            self.out_data["sum_euclidean_distance_hull"] = (
                self.out_data["m1_hull_sum_euclidean"]
                + self.out_data["m2_hull_sum_euclidean"]
            )
            
            self.out_data["total_movement_nose"] = self.out_data.eval(
                "movement_mouse_1_nose + movement_mouse_2_nose"
            )
            self.out_data["total_movement_tail_base"] = self.out_data.eval(
                "movement_mouse_1_tail_base + movement_mouse_2_tail_base"
            )
            
            self.out_data["total_movement_all_bodyparts_m2"] = self.out_data.eval(
                "movement_mouse_2_nose + movement_mouse_2_tail_base + movement_mouse_2_ear_left + "
                "movement_mouse_2_ear_right + movement_mouse_2_head + movement_mouse_2_body_center"
            )
            self.out_data["total_movement_all_bodyparts_both_mice"] = (
                self.out_data.eval("total_movement_all_bodyparts_m1 + total_movement_all_bodyparts_m2")
            )
        
        # Rolling window features
        print("Calculating rolling window features...")
        if self.paired:
            window_features = [
                "sum_euclidean_distance_hull", "total_movement_all_bodyparts_both_mice",
                "nose_to_nose_distance", "tail_to_tail_distance", "m1_hull_mean_euclidean",
                "m2_hull_mean_euclidean", "m1_hull_small_euclidean", "m2_hull_small_euclidean",
                "m1_hull_large_euclidean", "m2_hull_large_euclidean",
                "total_movement_all_bodyparts_both_mice", "movement_mouse_1_tail_base",
                "movement_mouse_2_tail_base", "movement_mouse_1_nose",
                "movement_nose_to_nose_distance"
            ]
        else:
            window_features = [
                "m1_hull_mean_euclidean", "m1_hull_small_euclidean", "m1_hull_large_euclidean",
                "movement_mouse_1_tail_base", "movement_mouse_1_nose"
            ]
        
        for window_feature in window_features:
            for window in self.roll_windows_values:
                col_name = f'{window_feature}_median_{window}'
                self.out_data[col_name] = (
                    self.out_data[window_feature]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = f'{window_feature}_mean_{window}'
                self.out_data[col_name] = (
                    self.out_data[window_feature]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
        
        # Deviation features
        print("Calculating deviation features...")
        if self.paired:
            deviation_features = [
                "total_movement_all_bodyparts_both_mice", "sum_euclidean_distance_hull",
                "m1_hull_small_euclidean", "m1_hull_large_euclidean", "m1_hull_mean_euclidean",
                "movement_mouse_1_nose", "movement_mouse_2_nose", "poly_area_1", "poly_area_2"
            ]
            rolling_deviation_features = [
                "total_movement_all_bodyparts_both_mice_mean", "sum_euclidean_distance_hull_mean",
                "m1_hull_small_euclidean_mean", "m1_hull_large_euclidean_mean",
                "m2_hull_small_euclidean_mean", "m2_hull_large_euclidean_mean",
                "m1_hull_mean_euclidean_mean", "m2_hull_mean_euclidean_mean",
                "movement_nose_to_nose_distance_mean"
            ]
        else:
            deviation_features = [
                "m1_hull_small_euclidean", "m1_hull_large_euclidean", "m1_hull_mean_euclidean",
                "movement_mouse_1_nose", "poly_area_1"
            ]
            rolling_deviation_features = [
                "m1_hull_small_euclidean_mean", "m1_hull_large_euclidean_mean",
                "m1_hull_mean_euclidean_mean"
            ]
        
        for deviation_feature in deviation_features:
            self.out_data[f'{deviation_feature}_deviation'] = (
                self.out_data[deviation_feature].mean()
                - self.out_data[deviation_feature]
            )
        
        for rolling_deviation_feature in rolling_deviation_features:
            for window in self.roll_windows_values:
                self.out_data[f'{rolling_deviation_feature}_deviation_{window}'] = (
                    self.out_data[f'{rolling_deviation_feature}_{window}'].mean() 
                    - self.out_data[f'{rolling_deviation_feature}_{window}']
                )
        
        # Percentile rank features
        print("Calculating percentile ranks...")
        self.out_data["movement_mouse_1_percentile_rank"] = self.out_data[
            "movement_mouse_1_nose"
        ].rank(pct=True)
        self.out_data["movement_mouse_1_deviation_percentile_rank"] = self.out_data[
            "movement_mouse_1_nose_deviation"
        ].rank(pct=True)
        
        if self.paired:
            self.out_data["movement_mouse_2_percentile_rank"] = self.out_data[
                "movement_mouse_2_nose"
            ].rank(pct=True)
            self.out_data["movement_mouse_2_deviation_percentile_rank"] = self.out_data[
                "movement_mouse_2_nose_deviation"
            ].rank(pct=True)
            self.out_data["movement_percentile_rank"] = self.out_data[
                "total_movement_nose"
            ].rank(pct=True)
            self.out_data["distance_percentile_rank"] = self.out_data[
                "nose_to_nose_distance"
            ].rank(pct=True)
            
            for window in self.roll_windows_values:
                col_name = f"total_movement_all_bodyparts_both_mice_mean_{window}"
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )
            
            for window in self.roll_windows_values:
                col_name = f"sum_euclidean_distance_hull_mean_{window}"
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )
        
        for window in self.roll_windows_values:
            col_name = f"m1_hull_mean_euclidean_mean_{window}"
            deviation_col_name = col_name + "_percentile_rank"
            self.out_data[deviation_col_name] = (
                self.out_data[col_name].mean() - self.out_data[col_name]
            )
        
        for window in self.roll_windows_values:
            col_name = f"m1_hull_small_euclidean_mean_{window}"
            deviation_col_name = col_name + "_percentile_rank"
            self.out_data[deviation_col_name] = (
                self.out_data[col_name].mean() - self.out_data[col_name]
            )
        
        for window in self.roll_windows_values:
            col_name = f"m1_hull_large_euclidean_mean_{window}"
            deviation_col_name = col_name + "_percentile_rank"
            self.out_data[deviation_col_name] = (
                self.out_data[col_name].mean() - self.out_data[col_name]
            )
        
        if self.paired:
            for window in self.roll_windows_values:
                col_name = f"m2_hull_mean_euclidean_mean_{window}"
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )
            
            for window in self.roll_windows_values:
                col_name = f"m2_hull_small_euclidean_mean_{window}"
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )
            
            for window in self.roll_windows_values:
                col_name = f"m2_hull_large_euclidean_mean_{window}"
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )
            
            for window in self.roll_windows_values:
                col_name = f"nose_to_nose_distance_mean_{window}"
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )
        
        # Probability scores
        print("Calculating pose probability scores...")
        if self.paired:
            all_p_columns = self.mouse_2_p_headers + self.mouse_1_p_headers
        else:
            all_p_columns = self.mouse_1_p_headers
        
        self.out_data["sum_probabilities"] = self.out_data[all_p_columns].sum(axis=1)
        self.out_data["sum_probabilities_deviation"] = (
            self.out_data["sum_probabilities"].mean()
            - self.out_data["sum_probabilities"]
        )
        self.out_data["sum_probabilities_deviation_percentile_rank"] = (
            self.out_data["sum_probabilities_deviation"].rank(pct=True)
        )
        self.out_data["sum_probabilities_percentile_rank"] = self.out_data[
            "sum_probabilities_deviation_percentile_rank"
        ].rank(pct=True)
        
        results = pd.DataFrame(
            self.count_values_in_range(
                data=self.out_data.filter(all_p_columns).values,
                ranges=np.array([[0.0, 0.1], [0.0, 0.5], [0.0, 0.75]]),
            ),
            columns=[
                "low_prob_detections_0.1",
                "low_prob_detections_0.5",
                "low_prob_detections_0.75",
            ],
        )
        self.out_data = pd.concat([self.out_data, results], axis=1)
        self.out_data = self.out_data.reset_index(drop=True).fillna(0)
        
        timer.stop_timer()
        print(f"Feature extraction complete (elapsed time: {timer.elapsed_time_str}s)")
        
        return self.out_data


# ==================== MAIN FUNCTIONS ====================

def main_snakemake(snakemake):
    """
    Snakemake-integrated entry point.
    
    Inputs:
        snakemake.input.processed_csv: Cleaned pose data
        snakemake.input.metadata: Video metadata JSON
        
    Outputs:
        snakemake.output.features_csv: Extracted features
        
    Params:
        snakemake.params.roll_windows: Rolling window sizes
        snakemake.params.body_parts: Body part names (for validation)
    """
    import json
    
    # Load metadata for FPS and px_per_mm
    with open(snakemake.input.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Use metadata values with fallbacks to global defaults
    global FPS, PX_PER_MM, ROLL_WINDOWS_VALUES,PAIRED
    FPS = metadata.get('fps', FPS)
    PX_PER_MM = metadata.get('px_per_mm', PX_PER_MM)
    ROLL_WINDOWS_VALUES = snakemake.params.roll_windows
    PAIRED= snakemake.params.experiment_type
    
    input_path = snakemake.input.processed_csv
    output_path = snakemake.output.features_csv
    
    print("=" * 70)
    print("FEATURE EXTRACTION (Snakemake Mode)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Pixels per mm: {PX_PER_MM}")
    print(f"  FPS: {FPS}")
    print(f"  Rolling windows: {ROLL_WINDOWS_VALUES}")
    print("=" * 70)
    
    # Read and reshape input data
    in_data, _ = read_sleap_csv(input_path)
    
    # Extract features
    extractor = FeatureExtractor(px_per_mm=PX_PER_MM, fps=FPS)
    out_data = extractor.extract_features(in_data, (PAIRED == "paired"))
    
    # Convert to float32 to save space
    out_data = out_data.astype(np.float32)
    
    # Save output
    print(f"\nSaving output to: {output_path}")
    out_data.to_csv(output_path, index=True)
    
    print("=" * 70)
    print(f"SUCCESS! Extracted {len(out_data.columns)} features from {len(out_data)} frames")
    print("=" * 70)


def main_standalone():
    """Standalone command-line entry point"""
    
    if len(sys.argv) != 3:
        print("Usage: python feature_extraction.py <input_csv> <output_csv>")
        print("\nExample:")
        print("  python feature_extraction.py input_data.csv output_features.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        sys.exit(1)
    
    print("=" * 70)
    print("FEATURE EXTRACTION (Standalone Mode)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Pixels per mm: {PX_PER_MM}")
    print(f"  FPS: {FPS}")
    print(f"  Rolling windows: {ROLL_WINDOWS_VALUES}")
    print("=" * 70)
    
    # Read and reshape input data
    in_data, paired = read_sleap_csv(input_path)
    
    # Extract features
    extractor = FeatureExtractor(px_per_mm=PX_PER_MM, fps=FPS)
    out_data = extractor.extract_features(in_data, paired)
    
    # Convert to float32 to save space and processing
    out_data = out_data.astype(np.float32)
    
    # Save output
    print(f"\nSaving output to: {output_path}")
    out_data.to_csv(output_path, index=True)
    
    print("=" * 70)
    print(f"SUCCESS! Extracted {len(out_data.columns)} features from {len(out_data)} frames")
    print("=" * 70)


if __name__ == "__main__":
    # Check if running via Snakemake
    if 'snakemake' in globals():
        main_snakemake(snakemake)
    else:
        main_standalone()