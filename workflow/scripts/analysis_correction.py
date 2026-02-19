#!/usr/bin/env python3
"""
Analysis Correction Script for Behavioral Pipeline

Uses manually scored data from the master scoring sheet to correct identity swaps
that occur in pose estimation and weren't corrected by the pose postprocessing script.

Compares predicted latency/time-in-nest with manual scores to detect swaps,
then corrects all feature values accordingly.

Only runs for paired animal samples (skipped for single animals).

Author: June (Molas Lab)
Adapted for Snakemake pipeline integration
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_manual_scores(excel_path: str, loom_day: str, animal_id: str) -> Dict:
    """
    Extract manual scores for a specific animal from the master scoring sheet.

    Parameters:
    -----------
    excel_path : str
        Path to the MasterScoringSheet Excel file
    loom_day : str
        LoomDay (refers to excel sheet name)
    animal_id : str or int
        Animal ID to extract (e.g., 218, 219)

    Returns:
    --------
    dict : {loom_number: (latency, time_in_nest)}
    """
    df = pd.read_excel(excel_path, sheet_name=loom_day, header=None)

    manual_scores = {}
    current_animal = None

    for idx, row in df.iterrows():
        # Check if this row starts a new animal section
        if pd.notna(row.iloc[0]) and str(animal_id) in str(row.iloc[0]):
            current_animal = str(animal_id)
        elif pd.notna(row.iloc[0]) and str(animal_id) not in str(row.iloc[0]):
            current_animal = None
            continue

        # If we're in the current animal's section and have a loom number
        if current_animal == str(animal_id) and pd.notna(row.iloc[1]):
            loom_num = row.iloc[1]
            if isinstance(loom_num, (int, float)) and not np.isnan(loom_num) and loom_num < 100:
                latency = row.iloc[17]  # Column R (index 17)
                time_in_nest = row.iloc[26]  # Column AA (index 26)
                manual_scores[int(loom_num)] = (latency, time_in_nest)

    return manual_scores


def is_missing(value) -> bool:
    """Check if a value is missing (NaN or -1)."""
    return pd.isna(value) or value == -1


def calculate_swap_decision(manual_a: Tuple, manual_b: Tuple,
                           pred_a: Tuple, pred_b: Tuple) -> bool:
    """
    Determine if identities should be swapped based on manual vs predicted comparison.

    Returns True if swap should be applied, False otherwise.
    """
    manual_a_lat, manual_a_time = manual_a
    manual_b_lat, manual_b_time = manual_b
    pred_a_lat, pred_a_time = pred_a
    pred_b_lat, pred_b_time = pred_b

    # Special handling: if latency is NaN, treat time as missing too
    manual_a_time_missing = is_missing(manual_a_time) or (is_missing(manual_a_lat) and manual_a_time == 0)
    manual_b_time_missing = is_missing(manual_b_time) or (is_missing(manual_b_lat) and manual_b_time == 0)

    # Check if data exists for each
    manual_a_has_data = not is_missing(manual_a_lat) or not manual_a_time_missing
    manual_b_has_data = not is_missing(manual_b_lat) or not manual_b_time_missing
    pred_a_has_data = not is_missing(pred_a_lat) or not is_missing(pred_a_time)
    pred_b_has_data = not is_missing(pred_b_lat) or not is_missing(pred_b_time)

    # SPECIAL CASE: If exactly one manual exists and exactly one predicted exists
    if manual_a_has_data and not manual_b_has_data and not pred_a_has_data and pred_b_has_data:
        return True
    elif not manual_a_has_data and manual_b_has_data and pred_a_has_data and not pred_b_has_data:
        return True
    elif manual_a_has_data and not manual_b_has_data and pred_a_has_data and not pred_b_has_data:
        return False
    elif not manual_a_has_data and manual_b_has_data and not pred_a_has_data and pred_b_has_data:
        return False

    # Calculate error without swap
    error_no_swap = 0.0
    count_no_swap = 0

    if not is_missing(manual_a_lat) and not is_missing(pred_a_lat):
        error_no_swap += abs(manual_a_lat - pred_a_lat)
        count_no_swap += 1
    if not is_missing(manual_a_time) and not is_missing(pred_a_time):
        error_no_swap += abs(manual_a_time - pred_a_time)
        count_no_swap += 1
    if not is_missing(manual_b_lat) and not is_missing(pred_b_lat):
        error_no_swap += abs(manual_b_lat - pred_b_lat)
        count_no_swap += 1
    if not is_missing(manual_b_time) and not is_missing(pred_b_time):
        error_no_swap += abs(manual_b_time - pred_b_time)
        count_no_swap += 1

    # Calculate error with swap
    error_with_swap = 0.0
    count_with_swap = 0

    if not is_missing(manual_a_lat) and not is_missing(pred_b_lat):
        error_with_swap += abs(manual_a_lat - pred_b_lat)
        count_with_swap += 1
    if not is_missing(manual_a_time) and not is_missing(pred_b_time):
        error_with_swap += abs(manual_a_time - pred_b_time)
        count_with_swap += 1
    if not is_missing(manual_b_lat) and not is_missing(pred_a_lat):
        error_with_swap += abs(manual_b_lat - pred_a_lat)
        count_with_swap += 1
    if not is_missing(manual_b_time) and not is_missing(pred_a_time):
        error_with_swap += abs(manual_b_time - pred_a_time)
        count_with_swap += 1

    if count_no_swap == 0 and count_with_swap == 0:
        return False

    return error_with_swap < error_no_swap


def correct_identity_swaps(animal_a_csv: str, animal_b_csv: str,
                           manual_excel: str, animal_a_id: str, animal_b_id: str,
                           experiment: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """
    Detect and correct identity swaps between two animals based on manual scoring.

    Returns:
    --------
    tuple : (corrected_a_df, corrected_b_df, swap_summary)
    """
    # Load data
    df_a = pd.read_csv(animal_a_csv)
    df_b = pd.read_csv(animal_b_csv)

    # Load manual scores
    manual_a = load_manual_scores(manual_excel, experiment, animal_a_id)
    manual_b = load_manual_scores(manual_excel, experiment, animal_b_id)

    # Create copies for correction
    corrected_a = df_a.copy()
    corrected_b = df_b.copy()

    # Add columns for swap tracking and errors
    corrected_a['Swap Before'] = 0
    corrected_b['Swap Before'] = 0
    corrected_a['Error Latency'] = np.nan
    corrected_b['Error Latency'] = np.nan
    corrected_a['Error Time in Nest'] = np.nan
    corrected_b['Error Time in Nest'] = np.nan

    swap_summary = []
    current_swap_state = False

    # Process each loom session
    for idx in range(len(df_a)):
        loom_num = df_a.loc[idx, 'Loom#']

        # Get manual scores for this loom
        manual_a_vals = manual_a.get(loom_num, (np.nan, np.nan))
        manual_b_vals = manual_b.get(loom_num, (np.nan, np.nan))

        # Get original predicted values
        pred_a_vals = (df_a.loc[idx, 'Latency to nest'], df_a.loc[idx, 'Time in nest'])
        pred_b_vals = (df_b.loc[idx, 'Latency to nest'], df_b.loc[idx, 'Time in nest'])

        # Determine if identities are swapped
        should_swap = calculate_swap_decision(manual_a_vals, manual_b_vals,
                                             pred_a_vals, pred_b_vals)

        new_swap_state = should_swap

        # Mark if parity changed to SWAPPED
        if new_swap_state and not current_swap_state:
            corrected_a.loc[idx, 'Swap Before'] = 1
            corrected_b.loc[idx, 'Swap Before'] = 1
        else:
            corrected_a.loc[idx, 'Swap Before'] = 0
            corrected_b.loc[idx, 'Swap Before'] = 0

        if should_swap:
            # Swap all feature values between the two animals
            for col in df_a.columns:
                if col not in ['AnimalID', 'Loom#', 'Sample']:
                    temp = corrected_a.loc[idx, col]
                    corrected_a.loc[idx, col] = corrected_b.loc[idx, col]
                    corrected_b.loc[idx, col] = temp

            swap_summary.append({
                'loom': int(loom_num),
                'swapped': True,
                'parity_changed': new_swap_state != current_swap_state,
                'reason': 'Swap reduced error or enforced alignment'
            })
        else:
            swap_summary.append({
                'loom': int(loom_num),
                'swapped': False,
                'parity_changed': False,
                'reason': 'No swap needed'
            })

        current_swap_state = new_swap_state

        # Calculate final errors
        final_pred_a_lat = corrected_a.loc[idx, 'Latency to nest']
        final_pred_a_time = corrected_a.loc[idx, 'Time in nest']
        final_pred_b_lat = corrected_b.loc[idx, 'Latency to nest']
        final_pred_b_time = corrected_b.loc[idx, 'Time in nest']

        # Error for animal A
        if not is_missing(manual_a_vals[0]) and not is_missing(final_pred_a_lat):
            corrected_a.loc[idx, 'Error Latency'] = abs(manual_a_vals[0] - final_pred_a_lat)
        else:
            corrected_a.loc[idx, 'Error Latency'] = 0

        if not is_missing(manual_a_vals[1]) and not is_missing(final_pred_a_time):
            corrected_a.loc[idx, 'Error Time in Nest'] = abs(manual_a_vals[1] - final_pred_a_time)
        else:
            corrected_a.loc[idx, 'Error Time in Nest'] = 0

        # Error for animal B
        if not is_missing(manual_b_vals[0]) and not is_missing(final_pred_b_lat):
            corrected_b.loc[idx, 'Error Latency'] = abs(manual_b_vals[0] - final_pred_b_lat)
        else:
            corrected_b.loc[idx, 'Error Latency'] = 0

        if not is_missing(manual_b_vals[1]) and not is_missing(final_pred_b_time):
            corrected_b.loc[idx, 'Error Time in Nest'] = abs(manual_b_vals[1] - final_pred_b_time)
        else:
            corrected_b.loc[idx, 'Error Time in Nest'] = 0

    # Save corrected files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    return corrected_a, corrected_b, swap_summary


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_animal_ids_from_sample(sample_name: str) -> Optional[Tuple[str, str]]:
    """
    Extract animal IDs from sample name for paired animals.
    Returns None for single-animal samples.
    """
    match = re.search(r'(\d{3})\+(\d{3})', sample_name)
    if match:
        return (match.group(1), match.group(2))
    return None


def extract_loom_session_from_sample(sample_name: str) -> str:
    """Extract loom session identifier from sample name."""
    match = re.search(r'[_]?[LD](\d+)', sample_name, re.IGNORECASE)
    if match:
        return f"L{match.group(1)}"
    return "L1"


# =============================================================================
# SNAKEMAKE INTEGRATION
# =============================================================================

def main_snakemake(snakemake):
    """Snakemake entry point."""
    print("=" * 70)
    print("ANALYSIS CORRECTION")
    print("=" * 70)

    # Get inputs
    analysis_done = snakemake.input.analysis_done
    scoring_sheet = snakemake.input.scoring_sheet

    # Get output
    output_dir = os.path.dirname(snakemake.output.done_flag)
    os.makedirs(output_dir, exist_ok=True)

    # Get sample info
    sample_name = snakemake.params.sample
    animal_ids = extract_animal_ids_from_sample(sample_name)

    # Skip single-animal samples
    if animal_ids is None:
        print(f"  Skipping single-animal sample: {sample_name}")
        with open(snakemake.output.done_flag, 'w') as f:
            f.write(f"Skipped - single animal sample\n")
            f.write(f"Sample: {sample_name}\n")
        print("=" * 70)
        print("SKIPPED (single animal)")
        print("=" * 70)
        return

    loom_session = extract_loom_session_from_sample(sample_name)

    print(f"  Sample: {sample_name}")
    print(f"  Animals: {animal_ids}")
    print(f"  Loom session: {loom_session}")

    # Find input analysis CSVs
    analysis_dir = os.path.dirname(analysis_done)
    animal_a_csv = os.path.join(analysis_dir, f"{sample_name}_Animal{animal_ids[0]}_analysis.csv")
    animal_b_csv = os.path.join(analysis_dir, f"{sample_name}_Animal{animal_ids[1]}_analysis.csv")

    if not os.path.exists(animal_a_csv) or not os.path.exists(animal_b_csv):
        print(f"  ERROR: Analysis files not found")
        print(f"    Expected: {animal_a_csv}")
        print(f"    Expected: {animal_b_csv}")
        raise FileNotFoundError("Analysis files not found")

    print(f"  Input A: {animal_a_csv}")
    print(f"  Input B: {animal_b_csv}")

    # Run correction
    corrected_a, corrected_b, swap_summary = correct_identity_swaps(
        animal_a_csv=animal_a_csv,
        animal_b_csv=animal_b_csv,
        manual_excel=scoring_sheet,
        animal_a_id=animal_ids[0],
        animal_b_id=animal_ids[1],
        experiment=loom_session,
        output_dir=output_dir
    )

    # Save corrected CSVs
    corrected_a_path = os.path.join(output_dir, f"{sample_name}_Animal{animal_ids[0]}_analysis_corrected.csv")
    corrected_b_path = os.path.join(output_dir, f"{sample_name}_Animal{animal_ids[1]}_analysis_corrected.csv")

    corrected_a.to_csv(corrected_a_path, index=False)
    corrected_b.to_csv(corrected_b_path, index=False)

    print(f"  Saved: {os.path.basename(corrected_a_path)}")
    print(f"  Saved: {os.path.basename(corrected_b_path)}")

    # Save swap summary as JSON
    swap_report_path = os.path.join(output_dir, f"{sample_name}_correction_report.json")
    swap_report = {
        'sample': sample_name,
        'animal_a': animal_ids[0],
        'animal_b': animal_ids[1],
        'loom_session': loom_session,
        'total_looms': len(swap_summary),
        'swaps_applied': int(sum(1 for s in swap_summary if s['swapped'])),
        'parity_changes': int(sum(1 for s in swap_summary if s['parity_changed'])),
        'details': swap_summary
    }

    with open(swap_report_path, 'w') as f:
        json.dump(swap_report, f, indent=2)

    print(f"  Saved: {os.path.basename(swap_report_path)}")

    # Print summary
    print(f"\n  Swap Summary:")
    print(f"    Total looms: {swap_report['total_looms']}")
    print(f"    Swaps applied: {swap_report['swaps_applied']}")
    print(f"    Parity changes: {swap_report['parity_changes']}")

    # Create completion flag
    with open(snakemake.output.done_flag, 'w') as f:
        f.write(f"Analysis correction completed for {sample_name}\n")
        f.write(f"Animals: {animal_ids}\n")
        f.write(f"Loom session: {loom_session}\n")
        f.write(f"Output files:\n")
        f.write(f"  - {os.path.basename(corrected_a_path)}\n")
        f.write(f"  - {os.path.basename(corrected_b_path)}\n")
        f.write(f"  - {os.path.basename(swap_report_path)}\n")

    print("=" * 70)
    print("CORRECTION COMPLETE")
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
        sys.exit(1)
