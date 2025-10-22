"""
Drift correction submodule for bioimage analysis pipeline.

This submodule contains utilities and functions for correcting sample drift
in time-lapse microscopy images.
"""

# Import key functions to make them available at package level
from .drift_correct_utils import drift_correction_score

__all__ = [
    'drift_correction_score'
]