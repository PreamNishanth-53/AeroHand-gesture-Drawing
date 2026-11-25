import numpy as np

def dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def hand_size(lm):
    """Wrist (0) to middle-MCP (9) — used to normalize thresholds."""
    return dist(lm[0], lm[9])
