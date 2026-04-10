"""
Shared pytest fixtures for the BIPHUB Pipeline Manager test suite.

All fixtures use synthetic in-memory data so no real image files are required.
"""

import sys
import os
import pytest
import numpy as np

# Make standard_code/python importable without installation.
_PYTHON_DIR = os.path.join(os.path.dirname(__file__), "..", "standard_code", "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


@pytest.fixture()
def rng():
    """Seeded numpy random generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture()
def zeros_5d():
    """5D all-zero TCZYX array (uint16) for shape/dtype checks."""
    return np.zeros((1, 1, 1, 16, 16), dtype=np.uint16)


@pytest.fixture()
def simple_circle_2d(rng):
    """
    64×64 uint8 image containing one filled bright circle on a dark background.
    The circle has a small dark 'hole' punched in its centre to test hole-filling.
    """
    img = np.zeros((64, 64), dtype=np.uint8)
    cy, cx = 32, 32
    for y in range(64):
        for x in range(64):
            d = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
            if d <= 20:
                img[y, x] = 180
    # Punch a dark hole in the centre
    img[28:36, 28:36] = 20
    return img


@pytest.fixture()
def labeled_mask_5d():
    """
    5D TCZYX uint16 mask (1, 1, 1, 20, 20) with:
    - label 1 in top-left 6×6 block
    - label 2 in bottom-right 6×6 block
    """
    mask = np.zeros((1, 1, 1, 20, 20), dtype=np.uint16)
    mask[0, 0, 0, 1:6, 1:6] = 1
    mask[0, 0, 0, 14:19, 14:19] = 2
    return mask
