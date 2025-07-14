import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def compute_distance_map(mask: np.ndarray, start: tuple[int, int], method: str = "geodesic") -> np.ndarray:
    """
    Compute distance map using different methods.
    
    Parameters:
        mask: Binary mask (non-zero is walkable)
        start: Tuple (y, x) of starting point
        method: One of ['euclidean', 'cityblock', 'chessboard', 'geodesic']
    
    Returns:
        Distance map as ndarray
    """
    if mask[start] == 0:
        raise ValueError(f"Starting point {start} is not inside a labeled object.")

    shape = mask.shape
    walkable = mask > 0
    coords = np.indices(shape).transpose(1, 2, 0).reshape(-1, 2)  # (N, 2) yx
    start_arr = np.array(start).reshape(1, 2)

    if method == "euclidean":
        dist_map = np.full(shape, np.inf)
        dist_map_flat = cdist(start_arr, coords, metric="euclidean").reshape(shape)
        dist_map[walkable] = dist_map_flat[walkable]
        return dist_map

    elif method == "cityblock":
        dist_map = np.full(shape, np.inf)
        dist_map_flat = cdist(start_arr, coords, metric="cityblock").reshape(shape)
        dist_map[walkable] = dist_map_flat[walkable]
        return dist_map

    elif method == "chessboard":
        dist_map = np.full(shape, np.inf)
        dist_map_flat = cdist(start_arr, coords, metric="chebyshev").reshape(shape)
        dist_map[walkable] = dist_map_flat[walkable]
        return dist_map

    elif method == "geodesic":
        mcp = MCP_Geometric(np.where(walkable, 1.0, np.inf), fully_connected=True)
        costs, _ = mcp.find_costs([start])
        costs[~walkable] = np.inf
        return costs

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'euclidean', 'cityblock', 'chessboard', 'geodesic'.")



def test_distance_methods():
    # Make a mask with some holes (obstacles)
    mask = np.ones((100, 100), dtype=bool)
    mask[0:70, 45:55] = 0  # vertical wall
    mask[60:80, 10:40] = 0  # horizontal blob
    start = (10, 10)

    methods = ['euclidean', 'cityblock', 'chessboard', 'geodesic']

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))

    for ax, method in zip(axes, methods):
        dist = compute_distance_map(mask, start, method=method)
        im = ax.imshow(dist, cmap="viridis", origin="upper")
        ax.set_title(method.capitalize())
        ax.scatter(start[1], start[0], color='red')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

test_distance_methods()