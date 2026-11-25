import cv2
import numpy as np


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _polygon_centroid(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return (0.0, 0.0)
    x = pts[:, 0].mean()
    y = pts[:, 1].mean()
    return (float(x), float(y))


def _fit_rotated_rect(pts):
    """Return (center, (w, h), angle) from cv2.minAreaRect."""
    rect = cv2.minAreaRect(np.asarray(pts, dtype=np.float32))
    (cx, cy), (w, h), angle = rect
    return (float(cx), float(cy)), (float(w), float(h)), float(angle)


def _regular_polygon(center, radius, n, rotation_deg=-90.0):
    """Return n vertices of a regular polygon."""
    cx, cy = center
    pts = []
    for i in range(n):
        ang = np.deg2rad(rotation_deg + i * 360.0 / n)
        pts.append((cx + radius * np.cos(ang), cy + radius * np.sin(ang)))
    return np.array(pts, dtype=np.float32)


# ----------------------------------------------------------------------
# shape detection + regularized drawing
# ----------------------------------------------------------------------
def detect_shape(points):
    """
    points : list of (x, y) from a stroke.
    Returns (shape_name, params) where params are already regularized
    (perfect shape) so the caller can draw them directly.
    """
    if len(points) < 8:
        return None, {}

    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

    hull = cv2.convexHull(pts)
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    if perimeter < 40 or area < 200:
        return None, {}

    (cx, cy), radius = cv2.minEnclosingCircle(hull)
    circle_area = np.pi * radius * radius
    circularity = area / (circle_area + 1e-6)

    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    n = len(approx)
    x, y, w, h = cv2.boundingRect(hull)
    aspect = w / float(h + 1e-6)

    # ---------- circle / oval ----------
    if circularity > 0.82:
        if 0.85 < aspect < 1.15:
            # Perfect circle
            r = int(round((w + h) / 4))
            return "Circle", {"center": (int(cx), int(cy)), "radius": r}
        else:
            # Perfect oval: fit ellipse
            ellipse = cv2.fitEllipse(hull)
            (ecx, ecy), (major, minor), ang = ellipse
            return "Oval", {
                "center": (int(ecx), int(ecy)),
                "axes": (int(major / 2), int(minor / 2)),
                "angle": float(ang),
            }

    # ---------- 3 sides ----------
    if n == 3:
        # Perfect triangle: use the 3 hull vertices as-is but draw closed
        pts3 = approx.reshape(-1, 2).astype(np.float32)
        return "Triangle", {"pts": pts3}

    # ---------- 4 sides ----------
    if n == 4:
        pts4 = _order_points(approx.reshape(-1, 2))
        (rcx, rcy), (rw, rh), rangle = _fit_rotated_rect(pts4)
        # sort so w >= h for a natural "width >= height" look
        if rw < rh:
            rw, rh = rh, rw
            rangle += 90.0

        # Decide which 4-gon
        tl, tr, br, bl = pts4
        side_top   = np.linalg.norm(tr - tl)
        side_bot   = np.linalg.norm(br - bl)
        side_left  = np.linalg.norm(bl - tl)
        side_right = np.linalg.norm(br - tr)
        sides = np.array([side_top, side_right, side_bot, side_left])
        avg = float(np.mean(sides)) + 1e-6
        variance = float(np.std(sides) / avg)

        # angle at each vertex
        def _ang(a, b, c):
            ba = a - b
            bc = c - b
            c_ = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return float(np.degrees(np.arccos(np.clip(c_, -1, 1))))
        angles = np.array([
            _ang(pts4[-1], pts4[0], pts4[1]),
            _ang(pts4[0],  pts4[1], pts4[2]),
            _ang(pts4[1],  pts4[2], pts4[3]),
            _ang(pts4[2],  pts4[3], pts4[0]),
        ])
        right_angles = bool(np.all(np.abs(angles - 90.0) < 25.0))

        # --- perfect square: side length = mean of sides, snapped ---
        if right_angles and variance < 0.18:
            side = int(round(avg))
            half = side / 2.0
            # keep the drawn orientation (rotate the square to match)
            th = np.deg2rad(rangle)
            cos_t = np.cos(th)
            sin_t = np.sin(th)
            dx = half
            dy = half
            def rot(px, py):
                return (rcx + px * cos_t - py * sin_t,
                        rcy + px * sin_t + py * cos_t)
            square_pts = np.array([
                rot(-dx, -dy),
                rot( dx, -dy),
                rot( dx,  dy),
                rot(-dx,  dy),
            ], dtype=np.float32)
            return "Square", {"pts": square_pts}

        # --- perfect rectangle ---
        if right_angles:
            half_w = rw / 2.0
            half_h = rh / 2.0
            th = np.deg2rad(rangle)
            cos_t = np.cos(th)
            sin_t = np.sin(th)
            def rot(px, py):
                return (rcx + px * cos_t - py * sin_t,
                        rcy + px * sin_t + py * cos_t)
            rect_pts = np.array([
                rot(-half_w, -half_h),
                rot( half_w, -half_h),
                rot( half_w,  half_h),
                rot(-half_w,  half_h),
            ], dtype=np.float32)
            return "Rectangle", {"pts": rect_pts}

        # --- rhombus: all sides = mean length, kept orientation ---
        if variance < 0.18:
            # build a rhombus using the diagonal from the rotated rect
            half_diag_x = rw / 2.0
            half_diag_y = rh / 2.0
            th = np.deg2rad(rangle)
            cos_t = np.cos(th)
            sin_t = np.sin(th)
            def rot(px, py):
                return (rcx + px * cos_t - py * sin_t,
                        rcy + px * sin_t + py * cos_t)
            rh_pts = np.array([
                rot(0.0, -half_diag_y),
                rot(half_diag_x, 0.0),
                rot(0.0, half_diag_y),
                rot(-half_diag_x, 0.0),
            ], dtype=np.float32)
            return "Rhombus", {"pts": rh_pts}

        # --- trapezoid: force top and bottom parallel, centered ---
        top_y = (pts4[0][1] + pts4[1][1]) / 2.0
        bot_y = (pts4[3][1] + pts4[2][1]) / 2.0
        top_x_l = pts4[0][0]
        top_x_r = pts4[1][0]
        bot_x_l = pts4[3][0]
        bot_x_r = pts4[2][0]
        trap_pts = np.array([
            (top_x_l, top_y),
            (top_x_r, top_y),
            (bot_x_r, bot_y),
            (bot_x_l, bot_y),
        ], dtype=np.float32)
        return "Trapezoid", {"pts": trap_pts}

    # ---------- regular polygons ----------
    if n in (5, 6, 8):
        center = (float(cx), float(cy))
        radius_f = float(radius)
        # find best orientation: rotate so vertices align with drawn hull
        # simple approach: use angle of first hull vertex
        first = hull.reshape(-1, 2)[0]
        dx = first[0] - cx
        dy = first[1] - cy
        base_ang = float(np.degrees(np.arctan2(dy, dx)))
        poly = _regular_polygon(center, radius_f, n, rotation_deg=base_ang)
        name = {5: "Pentagon", 6: "Hexagon", 8: "Octagon"}[n]
        return name, {"pts": poly}

    return None, {}


def draw_shape(layer, shape_name, params, color, thickness):
    """Draw the perfect shape onto 'layer'."""
    if shape_name == "Circle":
        cv2.circle(layer, params["center"], params["radius"],
                   color, thickness, lineType=cv2.LINE_AA)
        return

    if shape_name == "Oval":
        cv2.ellipse(layer, params["center"], params["axes"],
                    params.get("angle", 0.0), 0, 360,
                    color, thickness, lineType=cv2.LINE_AA)
        return

    # All other shapes come as pts arrays
    pts = params.get("pts")
    if pts is None:
        return
    pts_i = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(layer, [pts_i], True, color, thickness,
                  lineType=cv2.LINE_AA)
