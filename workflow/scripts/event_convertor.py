#!/usr/bin/env python3
"""
Event Convertor Script for Behavioral Pipeline

Converts event timestamps from multiple sources (scoring sheets, Bonsai logs, etc.)
into a frame-accurate event log using per-frame PTS data from the cropped video.

This script:
1. Loads per-frame PTS timestamps from crop_video.py output
2. Loads event definitions from configured sources (master scoring sheet, etc.)
3. Maps each event timestamp to the nearest video frame using PTS lookup
4. Computes signed frame_error (actual PTS of matched frame minus event time)
5. Exports an ordered event CSV for downstream analysis

The event CSV replaces the tight coupling between feature_analysis.py and the
master scoring sheet, providing a unified event interface that supports
multiple input formats.

Author: June (Molas Lab)
"""

import os
import sys
import re
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from bisect import bisect_left


# =============================================================================
# PTS LOADING
# =============================================================================

def load_pts(pts_csv_path: str) -> np.ndarray:
    """
    Load per-frame PTS timestamps from CSV.

    Args:
        pts_csv_path: Path to PTS CSV (columns: frame_idx, pts_seconds)

    Returns:
        1D numpy array of PTS values in seconds, indexed by frame number.
    """
    df = pd.read_csv(pts_csv_path)
    pts = df['pts_seconds'].to_numpy(dtype=np.float64)
    print(f"  Loaded {len(pts)} frame timestamps")
    print(f"  PTS range: {pts[0]:.6f}s — {pts[-1]:.6f}s")
    return pts


def timestamp_to_frame(pts: np.ndarray, timestamp_sec: float) -> Tuple[int, float]:
    """
    Map a timestamp (seconds) to the nearest frame index using PTS data.

    Uses binary search for O(log n) lookup.

    Args:
        pts: Sorted array of per-frame PTS values in seconds
        timestamp_sec: Event timestamp in seconds

    Returns:
        (frame_idx, frame_error) where:
            frame_idx: Index of the nearest frame
            frame_error: Signed error in seconds (pts[frame_idx] - timestamp_sec)
                         Positive = frame is after event, Negative = frame is before
    """
    n = len(pts)
    if n == 0:
        raise ValueError("PTS array is empty")

    # Binary search for insertion point
    idx = bisect_left(pts, timestamp_sec)

    # Handle boundary cases
    if idx == 0:
        return 0, float(pts[0] - timestamp_sec)
    if idx >= n:
        return n - 1, float(pts[-1] - timestamp_sec)

    # Compare with neighbors to find nearest
    before = idx - 1
    after = idx

    if abs(pts[before] - timestamp_sec) <= abs(pts[after] - timestamp_sec):
        best = before
    else:
        best = after

    error = float(pts[best] - timestamp_sec)
    return best, error


# =============================================================================
# EVENT SOURCE PARSERS
# =============================================================================

def parse_master_scoring_sheet(excel_path: str, loom_session: str,
                                animal_ids: tuple) -> List[Dict]:
    """
    Parse loom event times from the master scoring sheet.

    Extracts the same timing data as DataUtilities._extract_loom_times()
    in feature_analysis.py, but returns raw timestamps in seconds
    (no fps conversion — that happens via PTS lookup).

    Args:
        excel_path: Path to MasterScoringSheet Excel file
        loom_session: Sheet name / loom session identifier (e.g., 'L1')
        animal_ids: Tuple of animal IDs (e.g., ('218', '219'))

    Returns:
        List of event dicts with: event_id, event_type, time_start
    """
    df = pd.read_excel(excel_path, loom_session)

    # Fill NaN in timing column
    df['Unnamed: 4'] = df['Unnamed: 4'].fillna(0)

    # Find the row for this animal
    row_idx = -1
    for idx in range(len(df[loom_session])):
        cell_value = str(df.loc[idx, loom_session])
        if any(str(aid) in cell_value for aid in animal_ids):
            row_idx = idx
            break

    if row_idx < 0:
        print(f"  Warning: Could not find animal {animal_ids} in sheet {loom_session}")
        return []

    # Extract loom times (seconds)
    events = []
    loom_num = 0
    for j in range(6):  # Up to 6 looms
        try:
            time_sec = float(df['Unnamed: 4'][row_idx + j + 1])
            if time_sec > 0:
                loom_num += 1
                events.append({
                    'event_id': f"loom_{loom_num}",
                    'event_type': 'loom',
                    'time_start': time_sec,
                    'time_end': None
                })
        except (IndexError, ValueError):
            break

    # Remove trailing zero-time events
    while events and events[-1]['time_start'] == 0:
        events.pop()

    print(f"  Parsed {len(events)} loom events from scoring sheet")
    return events


def parse_bonsai_csv(csv_path: str, event_type_column: str = 'event_type',
                     time_column: str = 'timestamp',
                     end_time_column: Optional[str] = None) -> List[Dict]:
    """
    Parse event times from a Bonsai-generated CSV log.

    This is a generic parser for Bonsai event output. The CSV is expected
    to have at least a timestamp column and optionally an event type column.

    Args:
        csv_path: Path to Bonsai CSV file
        event_type_column: Column name for event type (default: 'event_type')
        time_column: Column name for timestamp in seconds (default: 'timestamp')
        end_time_column: Optional column name for event end time

    Returns:
        List of event dicts
    """
    if not os.path.exists(csv_path):
        print(f"  Warning: Bonsai CSV not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    events = []

    for idx, row in df.iterrows():
        event = {
            'event_id': f"bonsai_{idx+1}",
            'event_type': str(row.get(event_type_column, 'unknown')),
            'time_start': float(row[time_column]),
            'time_end': float(row[end_time_column]) if end_time_column and pd.notna(row.get(end_time_column)) else None
        }
        events.append(event)

    print(f"  Parsed {len(events)} events from Bonsai CSV")
    return events


# Registry of available parsers
EVENT_PARSERS = {
    'master_scoring_sheet': parse_master_scoring_sheet,
    'bonsai_csv': parse_bonsai_csv,
}


# =============================================================================
# EVENT CONVERSION
# =============================================================================

def collect_events(event_sources: List[Dict], sample_name: str,
                   animal_ids: tuple) -> List[Dict]:
    """
    Collect events from all configured sources.

    Args:
        event_sources: List of source configs from pipeline config, each with:
            - type: Parser name (key in EVENT_PARSERS)
            - path: Path to source file
            - Additional parser-specific kwargs
        sample_name: Sample name for context
        animal_ids: Animal IDs for context

    Returns:
        Sorted list of event dicts (by time_start)
    """
    all_events = []

    for source in event_sources:
        parser_type = source['type']
        source_path = source['path']

        if parser_type not in EVENT_PARSERS:
            print(f"  Warning: Unknown event parser type '{parser_type}', skipping")
            continue

        print(f"\n  Loading events from: {parser_type}")
        print(f"    Source: {source_path}")

        parser_fn = EVENT_PARSERS[parser_type]

        # Build kwargs from source config (exclude 'type' and 'path')
        kwargs = {k: v for k, v in source.items() if k not in ('type', 'path')}

        # Inject standard kwargs that parsers may need
        if parser_type == 'master_scoring_sheet':
            kwargs['excel_path'] = source_path
            kwargs['animal_ids'] = animal_ids
            # loom_session should be in the source config
        else:
            kwargs['csv_path'] = source_path

        events = parser_fn(**kwargs)
        all_events.extend(events)

    # Sort by time_start
    all_events.sort(key=lambda e: e['time_start'])

    print(f"\n  Total events collected: {len(all_events)}")
    return all_events


def convert_events_to_frames(events: List[Dict], pts: np.ndarray) -> List[Dict]:
    """
    Convert event timestamps to frame indices using PTS data.

    For each event, maps time_start (and optionally time_end) to the
    nearest frame, computing the signed error for each.

    Args:
        events: List of event dicts with time_start, time_end
        pts: Per-frame PTS array from cropped video

    Returns:
        List of event dicts augmented with:
            - frame_start: Nearest frame index for time_start
            - frame_end: Nearest frame index for time_end (or None)
            - frame_error_start: Signed error in seconds for start
            - frame_error_end: Signed error in seconds for end (or None)
    """
    converted = []

    for event in events:
        frame_start, error_start = timestamp_to_frame(pts, event['time_start'])

        frame_end = None
        error_end = None
        if event.get('time_end') is not None:
            frame_end, error_end = timestamp_to_frame(pts, event['time_end'])

        converted.append({
            'event_id': event['event_id'],
            'event_type': event['event_type'],
            'time_start': event['time_start'],
            'time_end': event.get('time_end'),
            'frame_start': frame_start,
            'frame_end': frame_end,
            'frame_error_start': error_start,
            'frame_error_end': error_end,
        })

    return converted


def export_events_csv(events: List[Dict], output_path: str):
    """
    Export frame-mapped events to CSV.

    Columns:
        event_id, event_type, time_start, time_end,
        frame_start, frame_end, frame_error_start, frame_error_end

    Args:
        events: List of converted event dicts
        output_path: Path to output CSV
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'event_id', 'event_type',
        'time_start', 'time_end',
        'frame_start', 'frame_end',
        'frame_error_start', 'frame_error_end'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = {k: event.get(k, '') for k in fieldnames}
            # Format floats
            for key in ['time_start', 'time_end', 'frame_error_start', 'frame_error_end']:
                if row[key] is not None and row[key] != '':
                    row[key] = f"{float(row[key]):.6f}"
                else:
                    row[key] = ''
            writer.writerow(row)

    print(f"  [OK] Events CSV: {output_path} ({len(events)} events)")


# =============================================================================
# SAMPLE NAME PARSING HELPERS
# =============================================================================

def extract_animal_ids_from_sample(sample_name: str) -> tuple:
    """
    Extract animal IDs from sample name.

    Handles patterns like:
    - "TRAP2-153+154_D1" -> ('153', '154')
    """
    match = re.search(r'(\d{3})\+(\d{3})', sample_name)
    if match:
        return (match.group(1), match.group(2))

    match = re.search(r'TRAP2?[-_\s]?(\d{3})', sample_name)
    if match:
        return (match.group(1),)

    return ('1',)


def extract_loom_session_from_sample(sample_name: str) -> str:
    """
    Extract loom session identifier from sample name.

    Handles patterns like:
    - "..._D1_..." -> "L1"
    - "...videoL1" -> "L1"
    """
    match = re.search(r'[_]?[LD](\d+)', sample_name, re.IGNORECASE)
    if match:
        return f"L{match.group(1)}"
    return "L1"


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main_snakemake(snakemake):
    """Snakemake entry point."""
    print("=" * 70)
    print("EVENT CONVERTOR")
    print("=" * 70)

    # Inputs
    pts_csv = snakemake.input.pts_csv
    metadata_json = snakemake.input.metadata

    # Output
    events_csv = snakemake.output.events_csv

    # Params
    sample_name = snakemake.params.sample
    event_sources_config = snakemake.params.event_sources

    # Derive animal IDs and loom session from sample name
    animal_ids = extract_animal_ids_from_sample(sample_name)
    loom_session = extract_loom_session_from_sample(sample_name)

    print(f"  Sample: {sample_name}")
    print(f"  Animals: {animal_ids}")
    print(f"  Loom session: {loom_session}")
    print(f"  PTS source: {pts_csv}")

    # Load PTS data
    print("\n[1/3] Loading frame timestamps...")
    pts = load_pts(pts_csv)

    # Load metadata for reference
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    print(f"  Video FPS (average): {metadata['fps']:.4f}")
    print(f"  Frame count: {metadata['frame_count']}")

    # Inject loom_session into scoring sheet sources that need it
    resolved_sources = []
    for source in event_sources_config:
        src = dict(source)  # copy
        if src['type'] == 'master_scoring_sheet' and 'loom_session' not in src:
            src['loom_session'] = loom_session
        resolved_sources.append(src)

    # Collect events from all sources
    print("\n[2/3] Collecting events from configured sources...")
    events = collect_events(resolved_sources, sample_name, animal_ids)

    if len(events) == 0:
        print("  Warning: No events found — writing empty CSV")

    # Convert to frames
    print("\n[3/3] Converting event timestamps to frame indices...")
    converted_events = convert_events_to_frames(events, pts)

    # Report
    for evt in converted_events:
        err_ms = evt['frame_error_start'] * 1000
        print(f"    {evt['event_id']:>10s}  t={evt['time_start']:8.3f}s  "
              f"→ frame {evt['frame_start']:>6d}  "
              f"(error: {err_ms:+.3f}ms)")

    # Export
    export_events_csv(converted_events, events_csv)

    print("\n" + "=" * 70)
    print("EVENT CONVERSION COMPLETE")
    print("=" * 70)


def main_cli():
    """Command-line entry point."""
    if len(sys.argv) < 4:
        print("Usage: event_convertor.py <pts_csv> <metadata_json> <output_events_csv> [event_sources_json]")
        print("\nevent_sources_json format:")
        print('  [{"type": "master_scoring_sheet", "path": "...", "loom_session": "L1"}, ...]')
        sys.exit(1)

    pts_csv = sys.argv[1]
    metadata_json = sys.argv[2]
    output_csv = sys.argv[3]

    if len(sys.argv) > 4:
        with open(sys.argv[4], 'r') as f:
            event_sources = json.load(f)
    else:
        print("No event sources specified")
        sys.exit(1)

    pts = load_pts(pts_csv)

    with open(metadata_json, 'r') as f:
        metadata = json.load(f)

    events = collect_events(event_sources, "cli_sample", ('1',))
    converted = convert_events_to_frames(events, pts)
    export_events_csv(converted, output_csv)


if __name__ == "__main__":
    try:
        snakemake
        main_snakemake(snakemake)
    except NameError:
        main_cli()
