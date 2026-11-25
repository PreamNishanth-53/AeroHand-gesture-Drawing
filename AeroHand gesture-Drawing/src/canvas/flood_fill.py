import cv2
import numpy as np


def flood_fill(layer, seed, color, tolerance=30):
    """
    Fill the region around 'seed' (x, y) in 'layer' with BGR 'color'.
    Uses a mask-based approach that fills only inside stroke boundaries.
    Returns True if a fill was performed.
    """
    h, w = layer.shape[:2]
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < w and 0 <= y < h):
        return False

    # Build a mask: only fill contiguous non-stroke area
    gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    # Threshold: anything brighter than small value = stroke
    stroke_mask = (gray > 20).astype(np.uint8) * 255

    # Flood-fill on a copy of stroke_mask to find the enclosed region
    mask = np.zeros((h + 2, w + 2), np.uint8)
    fill_mask = stroke_mask.copy()
    cv2.floodFill(fill_mask, mask, (x, y), 255,
                  loDiff=tolerance, upDiff=tolerance,
                  flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    region = mask[1:-1, 1:-1]

    if region.sum() == 0:
        return False

    # Apply color to the region
    color_arr = np.array(color, dtype=np.uint8)
    layer[region > 0] = color_arr
    return True
