#!/usr/bin/env python3
"""
SLEAP 1.4.1 Compatible CSV Export Script

This script provides CSV export functionality for SLEAP 1.4.1 (legacy version)
which doesn't support the --format analysis.csv command-line option.

Usage:
    python sleap_legacy_export.py <input_slp> <output_csv>
"""

import sys
import os
from pathlib import Path


def export_to_csv(slp_path: str, csv_path: str):
    """
    Export SLEAP predictions to CSV format (SLEAP 1.4.1 compatible).
    
    Args:
        slp_path: Path to .slp file
        csv_path: Path to output .csv file
    """
    try:
        # Import SLEAP modules
        from sleap import Labels
        from sleap.info.write_tracking_h5 import main as write_analysis
        
        print(f"Loading SLEAP file: {slp_path}")
        
        # Load labels
        video_callback = Labels.make_video_callback([os.path.dirname(slp_path)])
        labels = Labels.load_file(slp_path, video_search=video_callback)
        
        print(f"Loaded {len(labels)} labeled frames")
        print(f"Videos in project: {len(labels.videos)}")
        
        # Get the video (should be only one for single-video tracking)
        if len(labels.videos) == 0:
            raise ValueError("No videos found in SLEAP project")
        
        video = labels.videos[0]
        print(f"Exporting video: {video.backend.filename}")
        
        # Export to CSV using SLEAP 1.4.1 API
        # In SLEAP 1.4.1, csv export is done via write_analysis with csv=True
        write_analysis(
            labels,
            output_path=csv_path,
            labels_path=slp_path,
            all_frames=True,
            video=video,
            csv=True  # This is the key parameter for CSV output in SLEAP 1.4.1
        )
        
        print(f"Successfully exported to: {csv_path}")
        
    except ImportError as e:
        print(f"Error: Failed to import SLEAP modules. Is SLEAP installed?", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python sleap_legacy_export.py <input_slp> <output_csv>")
        print("\nExample:")
        print("  python sleap_legacy_export.py predictions.slp output.csv")
        sys.exit(1)
    
    slp_path = sys.argv[1]
    csv_path = sys.argv[2]
    
    # Check input exists
    if not os.path.exists(slp_path):
        print(f"Error: Input file not found: {slp_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Run export
    export_to_csv(slp_path, csv_path)


if __name__ == "__main__":
    main()
