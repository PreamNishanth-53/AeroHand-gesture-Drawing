import time
import cv2
import numpy as np
from config import settings
from src.core.hand_tracker import HandTracker
from src.core.gesture_recognizer import (
    is_index_only, is_ok_sign, is_pinch_pose, is_pinch_move, is_open_palm,
)
from src.core.smoother import EMA, LandmarkSmoother
from src.canvas.drawing_canvas import DrawingCanvas
from src.canvas.object_transform import ObjectTransform
from src.canvas.shape_snap import detect_shape, draw_shape
from src.ui.overlay import Overlay
from src.ui.toolbar import Toolbar


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def index_tip(lm):
    return lm[8]


def pinch_midpoint(lm):
    return ((lm[4][0] + lm[8][0]) / 2, (lm[4][1] + lm[8][1]) / 2)


def hand_distance(lm_a, lm_b):
    ax, ay = pinch_midpoint(lm_a)
    bx, by = pinch_midpoint(lm_b)
    return float(np.hypot(ax - bx, ay - by))


def pinch_spread_px(lm):
    return float(np.hypot(lm[8][0] - lm[4][0], lm[8][1] - lm[4][1]))


def combined_metric(lm_a, lm_b):
    """
    Hands closer                -> smaller
    Pinches closer (thumb-index)-> smaller
    Hands further               -> bigger
    Pinches wider               -> bigger
    """
    return hand_distance(lm_a, lm_b) + pinch_spread_px(lm_a) + pinch_spread_px(lm_b)


def clear_all(canvas, xform, point_smoother, lm_smoother, overlay=None):
    canvas.clear()
    xform.reset()
    point_smoother.reset()
    lm_smoother.reset()
    if overlay is not None:
        overlay.toast("Cleared", (255, 200, 100))


def snap_last_stroke(canvas, overlay):
    if not canvas.strokes:
        overlay.toast("Nothing to snap", (200, 200, 200))
        return
    last = canvas.strokes[-1]
    if not hasattr(last, "points") or len(last.points) < 8:
        overlay.toast("Stroke too short", (200, 200, 200))
        return
    name, params = detect_shape(last.points)
    if name is None:
        overlay.toast("Couldn't detect shape", (200, 200, 200))
        return
    canvas.replace_last_stroke_with_shape(name, params, draw_shape)
    overlay.toast(f"Snapped: {name}", (100, 255, 150))


def handle_toolbar_event(event, canvas, xform, overlay,
                         point_smoother, lm_smoother, toolbar):
    if event is None:
        return
    if event in ("colors_open", "brush_open", "shapes_open"):
        return
    if event == "clear":
        clear_all(canvas, xform, point_smoother, lm_smoother, overlay)
        return
    if event == "undo":
        if canvas.undo():
            overlay.toast("Undo", (200, 200, 255))
        return
    if event == "redo":
        if canvas.redo():
            overlay.toast("Redo", (200, 200, 255))
        return
    if event == "save":
        path = canvas.save_png()
        overlay.toast(f"Saved {path}", (150, 255, 150))
        return
    if event == "tool_pen":
        canvas.tool = "pen"
        overlay.toast("Pen", (220, 220, 220))
        return
    if event == "tool_highlight":
        canvas.tool = "highlighter"
        overlay.toast("Highlighter", (255, 255, 120))
        return
    if event == "tool_eraser":
        canvas.tool = "eraser"
        overlay.toast("Eraser", (200, 200, 200))
        return
    if event == "tool_fill":
        canvas.tool = "fill"
        overlay.toast("Fill - point + OK", (150, 200, 255))
        return
    if event == "shape_mode_toggle":
        overlay.toast(f"Snap mode: {'ON' if toolbar.snap_mode else 'OFF'}",
                      (150, 255, 150))
        return
    if event.startswith("color_"):
        try:
            idx = int(event.split("_", 1)[1])
        except ValueError:
            return
        canvas.color = settings.PALETTE[idx]
        overlay.toast("Color set", settings.PALETTE[idx])
        return
    if event.startswith("brush_"):
        try:
            idx = int(event.split("_", 1)[1])
        except ValueError:
            return
        canvas.thickness = settings.BRUSH_SIZES[idx]
        overlay.toast(f"Brush {settings.BRUSH_LABELS[idx]}", (220, 220, 220))
        return
    if event.startswith("shape_"):
        try:
            idx = int(event.split("_", 1)[1])
        except ValueError:
            return
        overlay.toast(f"Shape lock: {settings.SHAPES[idx]}", (150, 255, 150))
        return


# ----------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(settings.CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {settings.CAM_INDEX}.")
        cap.release()
        return

    cap.set(3, settings.FRAME_WIDTH)
    cap.set(4, settings.FRAME_HEIGHT)

    ok, probe = cap.read()
    if not ok:
        print("ERROR: Could not read from camera.")
        cap.release()
        return
    REAL_H, REAL_W = probe.shape[:2]
    print(f"[GestureDraw] Camera real size: {REAL_W} x {REAL_H}")

    tracker        = HandTracker()
    canvas         = DrawingCanvas(REAL_W, REAL_H)
    xform          = ObjectTransform(REAL_W, REAL_H)
    lm_smoother    = LandmarkSmoother(alpha=0.5)
    point_smoother = EMA(alpha=0.6)
    toolbar        = Toolbar(REAL_W, REAL_H)
    overlay        = Overlay()

    right_ok_latch = False
    left_ok_latch  = False
    right_move_latch = False
    last_pt = None

    two_hand_miss = 0
    TWO_HAND_GRACE = 3

    last_time = time.time()
    fps = 0.0

    win = "GestureDraw"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        if frame.shape[0] != REAL_H or frame.shape[1] != REAL_W:
            frame = cv2.resize(frame, (REAL_W, REAL_H))

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        hands, res = tracker.process(frame)
        tracker.draw(frame, res)

        right_hand = None
        left_hand = None
        for hh in hands:
            if hh["label"] == "Right":
                right_hand = hh
            elif hh["label"] == "Left":
                left_hand = hh

        right_lm = None
        if right_hand is not None:
            right_lm = lm_smoother.update("Right", right_hand["lm"])
        else:
            lm_smoother.forget("Right")

        left_lm = None
        if left_hand is not None:
            left_lm = lm_smoother.update("Left", left_hand["lm"])
        else:
            lm_smoother.forget("Left")

        gesture_label = "idle"
        cursor = None
        pointer_pt = None
        cursor2 = None
        info_line = ""

        # ========================================================
        # TWO-HAND SCALE
        #   Both hands must be in PINCH POSE:
        #     index extended + middle/ring/pinky down
        #     thumb can be near OR far
        #   Fist -> index down -> not pinch pose -> no trigger.
        # ========================================================
        both_hands_present = (right_lm is not None and left_lm is not None)

        if both_hands_present:
            right_ok_pose = is_pinch_pose(right_lm)
            left_ok_pose  = is_pinch_pose(left_lm)
            valid = right_ok_pose and left_ok_pose
        else:
            valid = False

        if valid:
            two_hand_miss = 0
            metric = combined_metric(right_lm, left_lm)
            active = xform.update_two_hand(True, metric)
            if active:
                gesture_label = "TWO-HAND SCALE"
                info_line = f"metric={metric:.0f}  scale={xform.scale:.2f}"
                cursor  = pinch_midpoint(right_lm)
                cursor2 = pinch_midpoint(left_lm)
        else:
            two_hand_miss += 1
            if two_hand_miss > TWO_HAND_GRACE:
                xform.end_two_hand_scale()

        # ========================================================
        # TOOLBAR POINTER (LEFT hand index only)
        # ========================================================
        hover_target = None
        if left_lm is not None and is_index_only(left_lm):
            pointer_pt = point_smoother.update(index_tip(left_lm))
            hover_target = toolbar.hit_test(pointer_pt)
        else:
            point_smoother.reset()

        # ========================================================
        # LEFT HAND OK -> click hovered tool OR snap last stroke
        # ========================================================
        if left_lm is not None:
            ok_left_now = is_ok_sign(left_lm)
            if ok_left_now and not left_ok_latch:
                left_ok_latch = True
                if hover_target is not None:
                    event = toolbar.click(hover_target)
                    handle_toolbar_event(event, canvas, xform, overlay,
                                         point_smoother, lm_smoother, toolbar)
                else:
                    snap_last_stroke(canvas, overlay)
            if not ok_left_now:
                left_ok_latch = False
        else:
            left_ok_latch = False

        # ========================================================
        # RIGHT HAND OK -> click hovered tool
        # ========================================================
        if right_lm is not None:
            ok_right_now = is_ok_sign(right_lm)
            if ok_right_now and not right_ok_latch:
                right_ok_latch = True
                if hover_target is not None:
                    event = toolbar.click(hover_target)
                    handle_toolbar_event(event, canvas, xform, overlay,
                                         point_smoother, lm_smoother, toolbar)
            if not ok_right_now:
                right_ok_latch = False
        else:
            right_ok_latch = False

        # ========================================================
        # RIGHT HAND PINCH -> MOVE (single-hand only)
        # ========================================================
        if (not xform.two_hand_active
                and not both_hands_present
                and right_lm is not None):
            move_now = is_pinch_move(right_lm)

            if move_now:
                pinch_pt = pinch_midpoint(right_lm)
                xform.move_to(pinch_pt)
                cursor = pinch_pt
                gesture_label = "move"
                canvas.lift_pen()
                last_pt = None
                right_move_latch = True
            else:
                if right_move_latch:
                    xform.release()
                    right_move_latch = False

        # ========================================================
        # RIGHT HAND DRAW
        # ========================================================
        if (not xform.two_hand_active
                and not right_move_latch
                and right_lm is not None):
            draw_now = is_index_only(right_lm)

            if draw_now:
                tip = index_tip(right_lm)
                tip_x, tip_y = int(tip[0]), int(tip[1])

                if last_pt is None:
                    last_pt = (tip_x, tip_y)
                    canvas.start_or_continue((tip_x, tip_y))
                else:
                    x0, y0 = last_pt
                    dist = float(np.hypot(tip_x - x0, tip_y - y0))
                    steps = max(1, int(dist / 3))
                    for i in range(1, steps + 1):
                        ix = int(x0 + (tip_x - x0) * i / steps)
                        iy = int(y0 + (tip_y - y0) * i / steps)
                        canvas.start_or_continue((ix, iy))
                    last_pt = (tip_x, tip_y)

                cursor = (tip_x, tip_y)
                gesture_label = "draw"
            else:
                was_drawing = (canvas.current_stroke is not None
                               and len(canvas.current_stroke.points) > 1)
                canvas.lift_pen()
                last_pt = None
                if was_drawing and toolbar.snap_mode:
                    last = canvas.strokes[-1]
                    if hasattr(last, "points") and len(last.points) >= 8:
                        name, params = detect_shape(last.points)
                        if name is not None:
                            canvas.replace_last_stroke_with_shape(
                                name, params, draw_shape)
                            overlay.toast(f"Auto-snap: {name}",
                                          (100, 255, 150))

            if is_open_palm(right_lm):
                clear_all(canvas, xform, point_smoother,
                          lm_smoother, overlay)
        elif right_lm is None:
            canvas.lift_pen()
            last_pt = None

        # ========================================================
        # RENDER
        # ========================================================
        warped, mask = xform.render(canvas.layer)
        frame[mask] = warped[mask]

        toolbar.render(frame, pointer_pt, hover_target)

        Overlay.draw_hud(frame, gesture_label, xform.scale,
                         canvas.tool, fps)
        if cursor is not None:
            Overlay.draw_cursor(frame, cursor, (0, 0, 255))
        if cursor2 is not None:
            Overlay.draw_cursor(frame, cursor2, (255, 0, 0))
        Overlay.draw_legend(frame)

        if info_line:
            cv2.putText(frame, info_line, (300, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        overlay.render_toasts(frame)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == settings.KEY_QUIT:
            break
        elif key == settings.KEY_CLEAR:
            clear_all(canvas, xform, point_smoother, lm_smoother, overlay)
        elif key == settings.KEY_UNDO:
            if canvas.undo():
                overlay.toast("Undo", (200, 200, 255))
        elif key == settings.KEY_REDO:
            if canvas.redo():
                overlay.toast("Redo", (200, 200, 255))
        elif key == settings.KEY_SAVE:
            path = canvas.save_png()
            overlay.toast(f"Saved {path}", (150, 255, 150))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
