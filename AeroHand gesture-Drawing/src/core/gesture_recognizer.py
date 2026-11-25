import numpy as np


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def fingers_up(lm):
    """
    Strict up-detection. A finger counts as 'up' only if its tip is
    clearly above its PIP by 8px and above its MCP by 15px.
    """
    up = []
    up.append(lm[4][0] > lm[3][0] + 6)   # thumb sideways
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        cond = (lm[tip][1] < lm[pip][1] - 8) and (lm[tip][1] < lm[mcp][1] - 15)
        up.append(cond)
    return up


def pinch_ratio(lm):
    d = _dist(lm[8], lm[4])
    hs = _dist(lm[0], lm[9]) + 1e-6
    return d / hs


def thumb_folded(lm):
    """Thumb tip near palm (not sticking out sideways)."""
    hs = _dist(lm[0], lm[9]) + 1e-6
    d = _dist(lm[4], lm[9])
    return (d / hs) < 0.9


def is_index_only(lm):
    """
    Strict draw pose: only the index extended, thumb folded, no pinch.
    """
    _, idx, mid, ring, pinky = fingers_up(lm)
    if not (idx and (not mid) and (not ring) and (not pinky)):
        return False
    if pinch_ratio(lm) <= 0.9:
        return False
    if not thumb_folded(lm):
        return False
    return True


def is_ok_sign(lm):
    _, idx, mid, ring, pinky = fingers_up(lm)
    return pinch_ratio(lm) < 0.55 and mid and ring and pinky


def is_pinch_pose(lm):
    """
    Two-hand scale trigger pose:
      - index extended
      - middle, ring, pinky all DOWN
      - thumb can be near the index (pinch closed) OR far (pinch open)
    Fist -> index down -> False.
    Open palm -> other fingers up -> False.
    """
    _, idx, mid, ring, pinky = fingers_up(lm)
    if not idx:
        return False
    if mid or ring or pinky:
        return False
    return True


def is_pinch(lm):
    """Legacy: strict pinch by thumb-index proximity."""
    return pinch_ratio(lm) < 0.55


def is_pinch_move(lm):
    """
    Single-hand right-hand move pinch:
      - thumb near index (closed pinch)
      - index extended
      - middle, ring, pinky down
    """
    _, idx, mid, ring, pinky = fingers_up(lm)
    if pinch_ratio(lm) >= 0.55:
        return False
    if not idx:
        return False
    if mid or ring or pinky:
        return False
    return True


def is_open_palm(lm):
    _, idx, mid, ring, pinky = fingers_up(lm)
    return idx and mid and ring and pinky and pinch_ratio(lm) > 0.7
