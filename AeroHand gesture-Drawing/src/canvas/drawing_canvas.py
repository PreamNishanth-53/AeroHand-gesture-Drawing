import os
import time
import cv2
import numpy as np
from config import settings
from src.canvas.flood_fill import flood_fill


class Stroke:
    def __init__(self, color, thickness, tool):
        self.color = color
        self.thickness = thickness
        self.tool = tool
        self.points = []


class DrawingCanvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.layer = np.zeros((h, w, 3), np.uint8)

        # Current stroke being drawn
        self.current_stroke = None
        self.prev_point = None

        # Completed strokes -> full history
        self.strokes = []
        self.redo_stack = []

        # Current tool settings
        self.color = settings.DRAW_COLOR
        self.thickness = settings.DRAW_THICKNESS
        self.tool = "pen"           # pen | highlighter | eraser

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def start_or_continue(self, point):
        if self.current_stroke is None:
            self.current_stroke = Stroke(self.color, self.thickness, self.tool)

        p = (int(point[0]), int(point[1]))
        if self.prev_point is None:
            self.prev_point = p
            self.current_stroke.points.append(p)
            return

        if self.tool == "eraser":
            cv2.line(self.layer, self.prev_point, p, (0, 0, 0),
                     settings.ERASER_THICKNESS)
        elif self.tool == "highlighter":
            overlay = self.layer.copy()
            cv2.line(overlay, self.prev_point, p, self.color,
                     self.thickness * 3)
            cv2.addWeighted(overlay, settings.HIGHLIGHTER_ALPHA,
                            self.layer, 1 - settings.HIGHLIGHTER_ALPHA,
                            0, self.layer)
        else:
            cv2.line(self.layer, self.prev_point, p, self.color, self.thickness)

        self.prev_point = p
        self.current_stroke.points.append(p)

    def lift_pen(self):
        if self.current_stroke is not None and len(self.current_stroke.points) > 1:
            self.strokes.append(self.current_stroke)
            self.redo_stack.clear()
        self.current_stroke = None
        self.prev_point = None

    # ------------------------------------------------------------------
    # Fill
    # ------------------------------------------------------------------
    def fill_at(self, point):
        flood_fill(self.layer, point, self.color)

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------
    def _redraw_all(self):
        self.layer[:] = 0
        for stroke in self.strokes:
            pts = stroke.points
            for i in range(1, len(pts)):
                if stroke.tool == "eraser":
                    cv2.line(self.layer, pts[i - 1], pts[i], (0, 0, 0),
                             settings.ERASER_THICKNESS)
                elif stroke.tool == "highlighter":
                    overlay = self.layer.copy()
                    cv2.line(overlay, pts[i - 1], pts[i], stroke.color,
                             stroke.thickness * 3)
                    cv2.addWeighted(overlay, settings.HIGHLIGHTER_ALPHA,
                                    self.layer, 1 - settings.HIGHLIGHTER_ALPHA,
                                    0, self.layer)
                else:
                    cv2.line(self.layer, pts[i - 1], pts[i],
                             stroke.color, stroke.thickness)

    def undo(self):
        if not self.strokes:
            return False
        self.redo_stack.append(self.strokes.pop())
        self._redraw_all()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.strokes.append(self.redo_stack.pop())
        self._redraw_all()
        return True

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
    def clear(self):
        self.layer[:] = 0
        self.strokes.clear()
        self.redo_stack.clear()
        self.current_stroke = None
        self.prev_point = None

    # ------------------------------------------------------------------
    # Shape snap on last stroke
    # ------------------------------------------------------------------
    def replace_last_stroke_with_shape(self, shape_name, params, draw_fn):
        """Remove last stroke and draw the perfect shape in its place."""
        if not self.strokes:
            return
        last = self.strokes.pop()
        self.redo_stack.clear()
        self._redraw_all()
        draw_fn(self.layer, shape_name, params, last.color, last.thickness)
        # Push a synthetic stroke record so undo still works
        synthetic = Stroke(last.color, last.thickness, "shape")
        synthetic.points = []  # has no raw points; it's a shape
        # store shape info on the stroke object
        synthetic.shape = (shape_name, params)  # type: ignore
        self.strokes.append(synthetic)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_png(self):
        os.makedirs("saved", exist_ok=True)
        name = f"saved/drawing_{int(time.time())}.png"
        cv2.imwrite(name, self.layer)
        return name
