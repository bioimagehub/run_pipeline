import numpy as np
def mock_drift_correction(image: np.ndarray, **kwargs) -> np.ndarray:
    """Mock drift correction function that returns the input image unchanged."""
    return image