#!/usr/bin/env python3
"""
Pose Post-Processing Script for SLEAP Tracking Data
Handles both single and paired animal tracking with identity swap correction.

For multi-animal tracking:
- Detects and corrects identity swaps during collisions
- Detects and corrects identity swaps across gaps
- Removes overlapping/collision frames
- Interpolates missing keypoints

For single animal:
- Fills missing frames
- Interpolates missing keypoints
- Handles spurious SLEAP tracks (removes tracks appearing in <5% of frames)

SLEAP Track Handling:
- Normalizes inconsistent track naming ("track_0" vs 0)
- Removes spurious tracks that SLEAP occasionally creates on first frame
- Uses experiment_type parameter from Snakemake as hint

Author: June (Molas Lab)
Updated: 2026-02-09 - Added robust track handling
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


# =============================================================================
# TRACK NORMALIZATION AND EXPERIMENT TYPE DETECTION
# =============================================================================

def normalize_track_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize track column to handle inconsistent SLEAP output formats.
    
    Handles:
    - String tracks ("track_0", "track_1") -> integers (0, 1)
    - Integer tracks (0, 1) -> keep as is
    - Mixed formats -> normalize to integers
    
    Args:
        df: DataFrame with 'track' column
        
    Returns:
        DataFrame with normalized integer track IDs
    """
    if 'track' not in df.columns:
        return df
    
    # Convert string tracks to integers
    # "track_0" -> 0, "track_1" -> 1, etc.
    def parse_track(val):
        if pd.isna(val):
            return val
        if isinstance(val, str):
            # Extract number from "track_X" format
            if val.startswith('track_'):
                return int(val.split('_')[1])
            # Handle other string formats
            try:
                return int(val)
            except ValueError:
                return val
        return val
    
    df['track'] = df['track'].apply(parse_track)
    
    return df


def detect_experiment_type(df: pd.DataFrame, expected_type: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
    """
    Detect whether this is single or paired animal tracking.
    
    Handles SLEAP quirk where single-animal videos may have spurious tracks.
    For single animals, if both track 0 and 1 exist, keeps only track 0.
    
    Args:
        df: DataFrame with pose data (should be normalized first)
        expected_type: Optional hint from Snakemake ("single" or "paired")
        
    Returns:
        Tuple of (experiment_type, cleaned_df)
        - experiment_type: "single" or "paired"
        - cleaned_df: DataFrame with spurious tracks removed if applicable
    """
    if 'track' not in df.columns:
        return "single", df
    
    # Get unique non-NaN tracks
    unique_tracks = df['track'].dropna().unique()
    num_unique = len(unique_tracks)
    
    print(f"  Found {num_unique} unique track(s): {sorted(unique_tracks)}")
    
    # If expected type is provided, use it as a strong hint
    if expected_type:
        print(f"  Expected type from Snakemake: {expected_type}")
        
        if expected_type == "single":
            if num_unique <= 1:
                return "single", df
            else:
                # Multiple tracks but single animal expected
                print(f"  Warning: Found {num_unique} tracks but single animal expected")
                
                track_counts = df['track'].value_counts()
                print(f"    Track counts: {dict(track_counts)}")
                
                # Simple rule: Keep only track 0, drop all others
                # SLEAP always creates track 0 first, so if track 1 exists in single mode, it's spurious
                print(f"    Keeping only track 0 (SLEAP's primary track)")
                tracks_to_remove = [t for t in unique_tracks if t != 0]
                if tracks_to_remove:
                    print(f"    Removing spurious track(s): {tracks_to_remove}")
                    for track in tracks_to_remove:
                        print(f"      - Track {track}: {track_counts[track]} frames")
                
                df_cleaned = df[df['track'].isna() | (df['track'] == 0)].copy()
                return "single", df_cleaned
                
        elif expected_type == "paired":
            if num_unique == 2:
                # Verify tracks are 0 and 1
                if sorted(unique_tracks) != [0, 1]:
                    print(f"  Warning: Expected tracks [0, 1] but found {sorted(unique_tracks)}")
                return "paired", df
            elif num_unique == 1:
                raise ValueError(f"Paired animals expected but found only {num_unique} track")
            else:
                raise ValueError(f"Paired animals expected but found {num_unique} tracks")
    
    # Fallback: determine from number of tracks (no Snakemake hint)
    # Without a hint, we can't distinguish spurious from real, so trust the data
    if num_unique <= 1:
        return "single", df
    elif num_unique == 2:
        print(f"  Two tracks detected without Snakemake hint - treating as paired")
        print(f"  If this is wrong, ensure Snakemake passes experiment_type parameter")
        return "paired", df
    else:
        raise ValueError(f"Unexpected number of tracks: {num_unique} (expected 1 or 2)")


# =============================================================================
# SINGLE ANIMAL PROCESSING
# =============================================================================

def process_single_animal(df: pd.DataFrame, body_parts: List[str], 
                         interpolation_limit: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Process single animal tracking data.
    
    Steps:
    1. Fill missing frame entries
    2. Interpolate missing coordinates
    
    Args:
        df: Raw SLEAP dataframe (should already be cleaned of spurious tracks)
        body_parts: List of body part names
        interpolation_limit: Max frames to interpolate across
        
    Returns:
        Tuple of (processed_df, report_dict)
    """
    print("\n[Single Animal Mode]")
    
    # Ensure frame_idx is int
    df['frame_idx'] = df['frame_idx'].astype(int)
    
    # Get full frame range
    min_frame = df['frame_idx'].min()
    max_frame = df['frame_idx'].max()
    frame_range = np.arange(min_frame, max_frame + 1)
    
    print(f"  Frame range: {min_frame} to {max_frame} ({len(frame_range)} frames)")
    
    # Reindex to fill missing frames
    df = df.set_index('frame_idx').reindex(frame_range).reset_index()
    
    if 'track' in df.columns:
        df['track'] =0
    # Count missing frames
    missing_frames = df[[f'{bp}.x' for bp in body_parts]].isna().all(axis=1).sum()
    print(f"  Missing frames filled: {missing_frames}")
    
    # Interpolate missing coordinates
    df, interpolated_intervals = interpolate_missing_data(df, body_parts, limit=interpolation_limit)
    
    # Create report
    report = {
        'interpolated_intervals': interpolated_intervals,
        'interpolation_limit': interpolation_limit
    }
    
    return df, report


def interpolate_missing_data(df: pd.DataFrame, body_parts: List[str],
                             limit: int = 5) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    Linearly interpolate missing keypoint coordinates.
    
    Args:
        df: Dataframe with pose data
        body_parts: List of body part names
        limit: Maximum number of consecutive NaNs to fill
        
    Returns:
        Tuple of (dataframe with interpolated values, list of interpolated intervals)
        Intervals are closed: [(start, end), ...] where both endpoints are included
    """
    interpolated_count = 0
    interpolated_intervals = []
    
    # Use first body part's x coordinate to track intervals (all body parts have same gaps)
    reference_col = f'{body_parts[0]}.x'
    if reference_col in df.columns:
        # Find NaN regions before interpolation
        is_nan = df[reference_col].isna()
        
        # Find consecutive NaN groups
        nan_groups = []
        in_gap = False
        gap_start = None
        
        for idx, is_missing in enumerate(is_nan):
            if is_missing and not in_gap:
                # Start of new gap
                gap_start = idx
                in_gap = True
            elif not is_missing and in_gap:
                # End of gap
                gap_end = idx - 1
                nan_groups.append((gap_start, gap_end))
                in_gap = False
        
        # Handle gap extending to end
        if in_gap:
            nan_groups.append((gap_start, len(df) - 1))
    
    # Interpolate all body parts
    for bp in body_parts:
        for suffix in ['.x', '.y', '.score']:
            col = f'{bp}{suffix}'
            if col in df.columns:
                before_na = df[col].isna().sum()
                # REMOVED limit_area='inside' to allow edge interpolation/extrapolation
                df[col] = df[col].interpolate(method='linear', limit=limit)
                after_na = df[col].isna().sum()
                interpolated_count += (before_na - after_na)
    
    # Determine which intervals were actually filled
    if reference_col in df.columns:
        is_nan_after = df[reference_col].isna()
        
        for gap_start, gap_end in nan_groups:
            gap_length = gap_end - gap_start + 1
            
            # Check if gap was filled (at least partially)
            gap_filled = not is_nan_after.iloc[gap_start:gap_end+1].all()
            
            if gap_filled:
                # Find actual filled region (may be smaller than original gap if limit exceeded)
                filled_start = None
                filled_end = None
                
                for idx in range(gap_start, gap_end + 1):
                    if not is_nan_after.iloc[idx]:
                        if filled_start is None:
                            filled_start = idx
                        filled_end = idx
                
                if filled_start is not None:
                    # Convert to frame indices
                    frame_start = df.iloc[filled_start]['frame_idx']
                    frame_end = df.iloc[filled_end]['frame_idx']
                    interpolated_intervals.append((int(frame_start), int(frame_end)))
    
    if interpolated_count > 0:
        print(f"  Interpolated {interpolated_count} missing values")
        if interpolated_intervals:
            print(f"  Interpolated intervals: {interpolated_intervals}")
    
    return df, interpolated_intervals


# =============================================================================
# MULTI-ANIMAL PROCESSING
# =============================================================================

def collision_detector(df0: pd.DataFrame, df1: pd.DataFrame, 
                      body_parts: List[str],
                      continuity_threshold: float = 150.0,
                      body_coords: List[str] = ['.x', '.y', '.score']) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Detect and remove colliding/overlapping identities.
    
    During collisions (animals close together), SLEAP may assign keypoints incorrectly.
    This function:
    1. Detects frames where animals are too close (< 3 separated body parts)
    2. Checks continuity with previous frame
    3. Removes the less continuous identity
    
    Args:
        df0, df1: DataFrames for the two animals
        body_parts: List of body part names
        continuity_threshold: Max allowed movement between frames (pixels)
        body_coords: Coordinate suffixes
        
    Returns:
        (df0_cleaned, df1_cleaned, collision_info)
    """
    print("\n  Detecting collisions...")
    
    # Extract coordinate arrays
    x0 = np.array([df0[f'{bp}.x'].values for bp in body_parts]).T
    y0 = np.array([df0[f'{bp}.y'].values for bp in body_parts]).T
    x1 = np.array([df1[f'{bp}.x'].values for bp in body_parts]).T
    y1 = np.array([df1[f'{bp}.y'].values for bp in body_parts]).T
    
    # Calculate distances between corresponding body parts
    distances = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # Count how many body parts are separated (> 50 pixels)
    separated_count = np.sum(distances > 50, axis=1)
    
    # Collision mask: < 3 separated body parts
    coll_mask = separated_count < 3
    coll_mask[0] = False  # First frame can't be evaluated
    
    # Compute framewise distances from previous frame
    x0_prev = np.roll(x0, 1, axis=0)
    y0_prev = np.roll(y0, 1, axis=0)
    x1_prev = np.roll(x1, 1, axis=0)
    y1_prev = np.roll(y1, 1, axis=0)
    
    # Correct wrapping
    x0_prev[0] = x0_prev[1]
    y0_prev[0] = y0_prev[1]
    x1_prev[0] = x1_prev[1]
    y1_prev[0] = y1_prev[1]
    
    def partwise_dist(ax, ay, bx, by):
        """Sum of Euclidean distances across body parts."""
        return np.nansum(np.sqrt((ax - bx)**2 + (ay - by)**2), axis=1)
    
    d00 = partwise_dist(x0, y0, x0_prev, y0_prev)
    d11 = partwise_dist(x1, y1, x1_prev, y1_prev)
    
    # Continuity masks
    cont00_bad = d00 > continuity_threshold
    cont11_bad = d11 > continuity_threshold
    
    # Removal conditions
    remove_both = (cont00_bad & cont11_bad & coll_mask)
    remove_0 = (d00 > d11) & coll_mask & (~remove_both)
    remove_1 = (d11 > d00) & coll_mask & (~remove_both)
    
    removed_indices = {
        'identity_0': np.where(remove_0)[0].tolist(),
        'identity_1': np.where(remove_1)[0].tolist(),
        'both': np.where(remove_both)[0].tolist()
    }
    
    # Remove bad frames
    cols = [f"{bp}{ax}" for bp in body_parts for ax in body_coords]
    df0.loc[remove_0 | remove_both, cols] = np.nan
    df1.loc[remove_1 | remove_both, cols] = np.nan
    
    total_removed = len(removed_indices['identity_0']) + len(removed_indices['identity_1']) + len(removed_indices['both'])
    print(f"    Removed {total_removed} collision frames")
    
    return df0, df1, removed_indices


def gap_identity_tracker(df0: pd.DataFrame, df1: pd.DataFrame,
                        body_parts: List[str]) -> List[int]:
    """
    Detect identity swaps across gaps where both animals are missing.
    
    When both animals disappear and reappear, they may swap identities.
    This detects such cases by comparing distances before/after the gap.
    
    Args:
        df0, df1: DataFrames for the two animals
        body_parts: List of body part names
        
    Returns:
        List of frame indices where swaps should be applied
    """
    print("\n  Detecting gap-based identity swaps...")
    
    all_frames = df0['frame_idx'].to_numpy()
    exist0 = df0.dropna(subset=[f"{bp}.x" for bp in body_parts])['frame_idx'].to_numpy()
    exist1 = df1.dropna(subset=[f"{bp}.x" for bp in body_parts])['frame_idx'].to_numpy()
    
    missing_both_mask = ~(np.isin(all_frames, exist0) | np.isin(all_frames, exist1))
    swapped_intervals = []
    
    # Split missing frames into consecutive gaps
    if missing_both_mask.any():
        double_gaps = np.split(
            all_frames[missing_both_mask],
            np.where(np.diff(all_frames[missing_both_mask]) != 1)[0] + 1
        )
        
        for gap in double_gaps:
            if len(gap) == 0:
                continue
            
            frame_start = gap[0] - 1
            frame_end = gap[-1] + 1
            
            # Get positions before and after gap
            last0 = df0[df0['frame_idx'] <= frame_start].tail(1)
            last1 = df1[df1['frame_idx'] <= frame_start].tail(1)
            next0 = df0[df0['frame_idx'] >= frame_end].head(1)
            next1 = df1[df1['frame_idx'] >= frame_end].head(1)
            
            if last0.empty or last1.empty or next0.empty or next1.empty:
                continue
            
            # Calculate distances for swapped vs non-swapped
            dist_no_swap = 0
            dist_swap = 0
            for bp in body_parts:
                for suffix in ['.x', '.y']:
                    last0_val = last0[f"{bp}{suffix}"].values[0]
                    last1_val = last1[f"{bp}{suffix}"].values[0]
                    next0_val = next0[f"{bp}{suffix}"].values[0]
                    next1_val = next1[f"{bp}{suffix}"].values[0]
                    
                    dist_no_swap += (next0_val - last0_val)**2 + (next1_val - last1_val)**2
                    dist_swap += (next0_val - last1_val)**2 + (next1_val - last0_val)**2
            
            if dist_swap < dist_no_swap:
                swapped_intervals.append(gap[-1])
    
    print(f"    Detected {len(swapped_intervals)} gap swaps")
    return swapped_intervals


def apply_cumulative_swaps(df0: pd.DataFrame, df1: pd.DataFrame,
                          swap_points: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply cumulative identity swaps at specified frame indices.
    
    Each swap point flips the identities. Swapping twice restores original.
    
    Args:
        df0, df1: DataFrames for the two animals
        swap_points: Frame indices where swaps occur
        
    Returns:
        (df0_swapped, df1_swapped)
    """
    if len(swap_points) == 0:
        return df0, df1
    
    df0 = df0.copy()
    df1 = df1.copy()
    
    swap_points = np.sort(np.unique(swap_points))
    frames = df0['frame_idx'].to_numpy()
    
    # Determine which frames should be flipped (odd number of swaps so far)
    flip_mask = np.cumsum(np.isin(frames, swap_points)) % 2 == 1
    
    # Get columns to swap (exclude frame_idx and track)
    cols_to_swap = [col for col in df0.columns if col not in ['frame_idx', 'track']]
    
    # Apply swap
    temp = df0.loc[flip_mask, cols_to_swap].copy()
    df0.loc[flip_mask, cols_to_swap] = df1.loc[flip_mask, cols_to_swap].values
    df1.loc[flip_mask, cols_to_swap] = temp.values
    
    print(f"    Applied swaps at {len(swap_points)} locations")
    
    return df0, df1


def process_paired_animals(df: pd.DataFrame, body_parts: List[str],
                          interpolation_limit: int = 5,
                          body_coords: List[str] = ['.x', '.y', '.score']) -> Tuple[pd.DataFrame, Dict]:
    """
    Process paired animal tracking data with identity swap correction.
    
    Args:
        df: Raw SLEAP dataframe (should have normalized tracks)
        body_parts: List of body part names
        interpolation_limit: Max frames to interpolate across
        body_coords: Coordinate suffixes
        
    Returns:
        (processed_df, swap_report)
    """
    print("\n[Paired Animal Mode]")
    
    # Ensure track is numeric
    df['track'] = pd.to_numeric(df['track'], errors='coerce')
    df['frame_idx'] = df['frame_idx'].astype(int)
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # Define frame range
    min_frame = df['frame_idx'].min()
    max_frame = df['frame_idx'].max()
    frame_range = np.arange(min_frame, max_frame + 1)
    
    print(f"  Frame range: {min_frame} to {max_frame} ({len(frame_range)} frames)")
    
    # Split into identity DataFrames
    identities = sorted(df['track'].dropna().unique())
    if len(identities) != 2:
        raise ValueError(f"Expected 2 identities, found {len(identities)}: {identities}")
    
    print(f"  Processing tracks: {identities}")
    
    tracks = {}
    for tid in identities:
        tdf = df[df['track'] == tid].set_index('frame_idx').reindex(frame_range).reset_index()
        tdf['track'] = tid
        tracks[tid] = tdf
    
    df0, df1 = tracks[identities[0]], tracks[identities[1]]
    
    # Remove collision frames
    df0, df1, collisions = collision_detector(df0, df1, body_parts, body_coords=body_coords)
    
    # Detect and fix gap swaps BEFORE interpolation
    swap_gaps = gap_identity_tracker(df0, df1, body_parts)
    df0, df1 = apply_cumulative_swaps(df0, df1, swap_gaps)
    
    # Interpolate missing data AFTER fixing swaps
    print("\n  Interpolating missing data...")
    df0, interp_intervals_0 = interpolate_missing_data(df0, body_parts, limit=interpolation_limit)
    df1, interp_intervals_1 = interpolate_missing_data(df1, body_parts, limit=interpolation_limit)
    
    # Merge back into single dataframe
    result_df = pd.concat([df0, df1], ignore_index=True).sort_values(['frame_idx', 'track'])
    
    swap_report = {
        'swap_events': swap_gaps,
        'collision_removals': collisions,
        'total_swaps': len(swap_gaps),
        'interpolated_intervals': {
            'identity_0': interp_intervals_0,
            'identity_1': interp_intervals_1
        },
        'interpolation_limit': interpolation_limit
    }
    
    return result_df, swap_report


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main(snakemake):
    """
    Main workflow for Snakemake integration.
    
    Inputs:
        snakemake.input.raw_csv: SLEAP predictions CSV
        snakemake.input.metadata: Video metadata JSON
        
    Outputs:
        snakemake.output.processed_csv: Cleaned pose data
        snakemake.output.swap_report: JSON report of corrections
        
    Params:
        snakemake.params.body_parts: List of body part names
        snakemake.params.interpolation_limit: Max frames to interpolate
        snakemake.params.experiment_type: "single" or "paired" (from Snakemake)
    """
    print("="*70)
    print("POSE POST-PROCESSING")
    print("="*70)
    
    # Load data
    df = pd.read_csv(snakemake.input.raw_csv)
    
    # Handle old SLEAP versions (.p -> .score)
    df.rename(columns=lambda x: x.replace('.p', '.score'), inplace=True)
    
    print(f"\nLoaded {len(df)} rows from CSV")
    
    # Load metadata
    with open(snakemake.input.metadata, 'r') as f:
        metadata = json.load(f)
    
    body_parts = snakemake.params.body_parts
    interpolation_limit = snakemake.params.interpolation_limit
    
    # Get expected experiment type from Snakemake
    expected_type = getattr(snakemake.params, 'experiment_type', None)
    if expected_type:
        print(f"\nExpected experiment type (from Snakemake): {expected_type}")
    
    # Step 1: Normalize track column to handle inconsistent naming
    print("\nNormalizing track naming...")
    df = normalize_track_column(df)
    
    # Step 2: Detect experiment type and remove spurious tracks
    print("\nDetecting experiment type...")
    experiment_type, df = detect_experiment_type(df, expected_type)
    print(f"Final experiment type: {experiment_type}")
    
    # Step 3: Process based on experiment type
    if experiment_type == "single":
        processed_df, single_report = process_single_animal(df, body_parts, interpolation_limit)
        swap_report = {
            'swap_events': [], 
            'mode': 'single_animal', 
            'experiment_type': experiment_type,
            'interpolated_intervals': single_report['interpolated_intervals'],
            'interpolation_limit': single_report['interpolation_limit']
        }
    elif experiment_type == "paired":
        processed_df, swap_report = process_paired_animals(df, body_parts, interpolation_limit)
        swap_report['mode'] = 'paired_animals'
        swap_report['experiment_type'] = experiment_type
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Save outputs
    print("\nSaving outputs...")
    processed_df.to_csv(snakemake.output.processed_csv, index=False)
    print(f"  [OK] Processed poses: {snakemake.output.processed_csv}")

    with open(snakemake.output.swap_report, 'w') as f:
        json.dump(convert_to_serializable(swap_report), f, indent=2)
    # print(f"  [OK] Swap report: {snakemake.output.swap_report}")
    
    print("\n" + "="*70)
    print("POST-PROCESSING COMPLETE")
    print(f"  Mode: {experiment_type}")
    if experiment_type == "paired":
        print(f"  Identity swaps corrected: {swap_report['total_swaps']}")
    print("="*70)


if __name__ == '__main__':
    main(snakemake)
