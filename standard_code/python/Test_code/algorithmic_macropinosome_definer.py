def segment_algorithmic(
    crop_2d: np.ndarray,
    center_y: int,
    center_x: int,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Segment a macropinosome using the BIPHUB macropinosome algorithm.

    Args:
        crop_2d:   (H, W) float or integer image array.
        center_y:  Row index of the seed point.
        center_x:  Column index of the seed point.
        tolerance: ±tolerance around the centre pixel value for the threshold.

    Returns:
        (H, W) float32 binary mask; 1.0 inside the macropinosome.
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label
    from skimage.morphology import reconstruction

    crop_f = crop_2d.astype(np.float32)
    H, W = crop_f.shape

    seed = crop_f.copy()
    seed[1:-1, 1:-1] = crop_f.max()
    filled = reconstruction(seed, crop_f, method="erosion").astype(np.float32)

    cy = int(np.clip(center_y, 0, H - 1))
    cx = int(np.clip(center_x, 0, W - 1))
    center_val = float(filled[cy, cx])

    binary = (filled >= center_val - tolerance) & (filled <= center_val + tolerance)
    binary = binary_fill_holes(binary)

    labeled = label(binary.astype(np.uint8), connectivity=1)
    comp_id = int(labeled[cy, cx])
    if comp_id == 0:
        logger.warning(
            "No component at seed (%d, %d); returning empty mask.", cy, cx
        )
        return np.zeros((H, W), dtype=np.float32)

    return (labeled == comp_id).astype(np.float32)
