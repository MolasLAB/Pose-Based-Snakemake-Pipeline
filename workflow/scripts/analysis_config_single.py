#!/usr/bin/env python3
"""
Analysis Configuration for Single Animal Feature Analysis Pipeline

This module defines the CalculatedFeatures configuration dictionary for
single-animal experiments. It excludes all pair-only features (midline angle,
nose-to-nose distance, nose-to-tail distance) while retaining all individual
behavioral metrics.

Author: June (Molas Lab)

Usage:
    from analysis_config_single import build_calculated_features_single

    features_config = build_calculated_features_single(fps=29.595, pixels_per_cm=32)
"""

import numpy as np


def build_calculated_features_single(fps: float, pixels_per_cm: float) -> dict:
    """
    Build the CalculatedFeatures configuration dictionary for single animals.

    This dictionary defines all features to be calculated during analysis.
    Each entry specifies:
        - func: Name of the method to call on AnimalBehaviorAnalyzer
        - kwargs: Arguments to pass to the function (including window definitions)
        - post: Post-processing function to apply to the result
        - multi: Whether the function returns multiple values
        - subfeatures: Names for each value if multi=True

    Window definitions use timepoint references:
        - frame_start_point: Reference timepoint ("loomtime", "latency", "time_in_nest")
        - frame_start_offset: Offset from start point (negative = before)
        - frame_end_point: Reference timepoint for end (defaults to start_point)
        - frame_end_offset: Offset from end point

    Args:
        fps: Frames per second of the video
        pixels_per_cm: Pixel to centimeter conversion factor

    Returns:
        Dictionary of feature configurations
    """

    # Post-processing function factories
    def frames_to_seconds(x):
        """Convert frame count to seconds"""
        if x is not None and x >= 0:
            return x / fps
        return x

    def pixels_to_cm_per_sec(x):
        """Convert pixel velocity to cm/s"""
        if x is not None and x >= 0:
            return x * fps / pixels_per_cm
        return x

    def pixels_to_cm(x):
        """Convert pixel distance to cm"""
        if x is not None and x >= 0:
            return x / pixels_per_cm
        return x

    def sum_values(x):
        """Sum array values"""
        return np.sum(x)

    # Feature configuration dictionary (single animal only)
    calculated_features = {
        # =====================================================================
        # TIMEPOINT-DEFINING FEATURES (calculate first - others depend on these)
        # =====================================================================

        "Freeze Start & Freeze Length": {
            "func": "FreezingAfterLoom",
            "post": frames_to_seconds,
            "multi": True,
            "subfeatures": ["Freeze start", "Freeze length"]
        },

        "Latency to nest": {
            "func": "LatencyToNest",
            "post": frames_to_seconds,
            "multi": False
        },

        "Time in nest": {
            "func": "TimeInNest",
            "post": frames_to_seconds,
            "multi": False
        },

        # =====================================================================
        # FREEZING FEATURES
        # =====================================================================

        "Freeze Percent (2s window starting at loom)": {
            "func": "FreezePercentageAfterLoom",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int(2 * fps)
                }
            },
            "post": None,
            "multi": False
        },

        "Freeze Percent (window over whole loom)": {
            "func": "FreezePercentageAfterLoom",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int(7.5 * fps)
                }
            },
            "post": None,
            "multi": False
        },

        # =====================================================================
        # ANGLE TO LOOM FEATURES
        # =====================================================================

        "Angle to loom(during loom)": {
            "func": "ROI_head_angle",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int(7.5 * fps)
                }
            },
            "post": None,
            "multi": False
        },

        "Angle to loom(0.2 seconds after loom)": {
            "func": "ROI_head_angle",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int(0.2 * fps)
                }
            },
            "post": None,
            "multi": False
        },

        "Angle to loom(10s before loom)": {
            "func": "ROI_head_angle",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": 0,
                    "frame_start_offset": -int(10 * fps)
                }
            },
            "post": None,
            "multi": False
        },

        "Facing loom percent": {
            "func": "ROI_head_facing",
            "post": None,
            "multi": False
        },

        # =====================================================================
        # VELOCITY FEATURES
        # =====================================================================

        "Max velocity(start loom - nest entry)": {
            "func": "MaxVelocityAfterLoom",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_point": "latency",
                    "frame_end_offset": 4,
                    "frame_start_offset": -4
                }
            },
            "post": pixels_to_cm_per_sec,
            "multi": False
        },

        "Max velocity(10s before loom)": {
            "func": "MaxVelocityAfterLoom",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_point": "loomtime",
                    "frame_end_offset": 4,
                    "frame_start_offset": -int(10 * fps) - 4
                }
            },
            "post": pixels_to_cm_per_sec,
            "multi": False
        },

        "Max velocity to nest(start loom - nest entry)": {
            "func": "MaxVelocityToNestAfterLoom",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_point": "latency",
                    "frame_end_offset": 4,
                    "frame_start_offset": -4
                }
            },
            "post": pixels_to_cm_per_sec,
            "multi": False
        },

        "Max velocity to nest(10s before loom)": {
            "func": "MaxVelocityToNestAfterLoom",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_point": "loomtime",
                    "frame_end_offset": 4,
                    "frame_start_offset": -int(10 * fps) - 4
                }
            },
            "post": pixels_to_cm_per_sec,
            "multi": False
        },

        # =====================================================================
        # DATA QUALITY FEATURES (Missing frame detection)
        # =====================================================================

        "Missing Body Center Frames": {
            "func": "missing_value_estimate",
            "kwargs": {
                "body_part": "body_center",
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int((7.5 + 10) * fps),
                    "frame_start_offset": int(-10 * fps)
                }
            },
            "post": sum_values,
            "multi": False
        },

        "Missing Head Frames": {
            "func": "missing_value_estimate",
            "kwargs": {
                "body_part": "head",
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int((7.5 + 10) * fps),
                    "frame_start_offset": int(-10 * fps)
                }
            },
            "post": sum_values,
            "multi": False
        },

        "Missing Left Ear Frames": {
            "func": "missing_value_estimate",
            "kwargs": {
                "body_part": "ear_left",
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int((7.5 + 10) * fps),
                    "frame_start_offset": int(-10 * fps)
                }
            },
            "post": sum_values,
            "multi": False
        },

        "Missing Nose Frames": {
            "func": "missing_value_estimate",
            "kwargs": {
                "body_part": "nose",
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int((7.5 + 10) * fps),
                    "frame_start_offset": int(-10 * fps)
                }
            },
            "post": sum_values,
            "multi": False
        },

        "Missing Right Ear Frames": {
            "func": "missing_value_estimate",
            "kwargs": {
                "body_part": "ear_right",
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int((7.5 + 10) * fps),
                    "frame_start_offset": int(-10 * fps)
                }
            },
            "post": sum_values,
            "multi": False
        },

        "Missing Tail Base Frames": {
            "func": "missing_value_estimate",
            "kwargs": {
                "body_part": "tail_base",
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int((7.5 + 10) * fps),
                    "frame_start_offset": int(-10 * fps)
                }
            },
            "post": sum_values,
            "multi": False
        },

        "Missing Identity Frames": {
            "func": "total_missing_parts",
            "kwargs": {
                "window": {
                    "frame_start_point": "loomtime",
                    "frame_end_offset": int((7.5 + 10) * fps),
                    "frame_start_offset": int(-10 * fps)
                }
            },
            "post": sum_values,
            "multi": False
        }
    }

    return calculated_features


# Default configuration for quick testing
DEFAULT_FPS = 29.595
DEFAULT_PIXELS_PER_CM = 32

if __name__ == "__main__":
    # Test: print feature names
    config = build_calculated_features_single(DEFAULT_FPS, DEFAULT_PIXELS_PER_CM)
    print(f"Defined {len(config)} single-animal features:")
    for name in config:
        print(f"  - {name}")
