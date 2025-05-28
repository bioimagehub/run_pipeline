from typing import Callable, List, Optional, Tuple, Union, Literal, Dict, Any

SysExitStep = Literal[
    "median_filter",
    "background_subtract",
    "apply_threshold",
    "thresholding",
    "remove_small_labels",
    "remove_edges",
    "fill_holes",
    "watershed",
    "tracking",
    "generate_rois",
]

print(dict(SysExitStep))
