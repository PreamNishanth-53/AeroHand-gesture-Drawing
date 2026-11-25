import cv2
import numpy as np
from config import settings


BUTTONS = [
    # id, label, glyph-kind
    ("colors",     "Color",  "palette"),
    ("brush",      "Size",   "brush"),
    ("pen",        "Pen",    "pen"),
    ("highlight",  "Mark",   "highlight"),
    ("eraser",     "Erase",  "eraser"),
    ("fill",       "Fill",   "fill"),
    ("shape_mode", "Snap",   "snap"),
    ("shape_lock", "Shape",  "shape"),
    ("undo",       "Undo",   "undo"),
    ("redo",       "Redo",   "redo"),
    ("save",       "Save",   "save"),
    ("clear",      "Clear",  "clear"),
]


class Toolbar:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.x0 = settings.TOOLBAR_LEFT_MARGIN
        self.y0 = settings.TOOLBAR_TOP_MARGIN
        self.bw = settings.TOOLBAR_BTN_W
        self.bh = settings.TOOLBAR_BTN_H
        self.gap = settings.TOOLBAR_GAP

        self.open_panel = None
        self.selected_color_idx = 6
        self.selected_brush_idx = 1
        self.selected_shape = "Circle"
        self.snap_mode = False

        self.rects = {}
        self._build_layout()

    def _build_layout(self):
        x = self.x0
        y = self.y0
        for bid, _, _ in BUTTONS:
            self.rects[bid] = (x, y, self.bw, self.bh)
            x += self.bw + self.gap

    # ---------------------------------------------------------------
    # Drawing helpers
    # ---------------------------------------------------------------
    @staticmethod
    def _rounded(img, pt1, pt2, color, radius=8, thickness=-1, alpha=1.0):
        x1, y1 = pt1
        x2, y2 = pt2
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            img[:] = overlay
        if thickness > 0:
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    def _draw_button(self, img, rect, label, glyph, hovered, active):
        x, y, w, h = rect
        base = (40, 40, 40)
        if active:
            base = (70, 110, 160)
        if hovered:
            base = (80, 80, 80)
        self._rounded(img, (x, y), (x + w, y + h), base,
                      radius=settings.TOOLBAR_RADIUS, alpha=0.9)
        border = (200, 220, 255) if hovered else (110, 110, 110)
        self._rounded(img, (x, y), (x + w, y + h), border,
                      radius=settings.TOOLBAR_RADIUS, thickness=1)

        # Icon area: top 38 px
        icon_cx = x + w // 2
        icon_cy = y + 20
        self._draw_icon(img, glyph, icon_cx, icon_cy, hovered)

        # Label area: bottom
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + h - 8
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (240, 240, 240) if not hovered else (255, 255, 200),
                    1, cv2.LINE_AA)

    def _draw_icon(self, img, kind, cx, cy, hovered):
        col = (90, 200, 255) if hovered else (230, 230, 230)
        r = 11

        if kind == "palette":
            cv2.circle(img, (cx, cy), r, col, 1)
            cv2.circle(img, (cx - 4, cy - 3), 2, col, -1)
            cv2.circle(img, (cx + 4, cy - 3), 2, col, -1)
            cv2.circle(img, (cx,     cy + 4), 2, col, -1)
        elif kind == "brush":
            cv2.line(img, (cx - r + 2, cy + r - 2),
                     (cx + r - 5, cy - r + 5), col, 2)
            cv2.circle(img, (cx - r + 2, cy + r - 2), 2, col, -1)
        elif kind == "pen":
            pts = np.array([[cx - r + 3, cy + r - 2],
                            [cx + r - 3, cy - r + 2],
                            [cx + r - 3, cy - r + 6],
                            [cx - r + 5, cy + r - 2]], np.int32)
            cv2.polylines(img, [pts], True, col, 2)
        elif kind == "highlight":
            cv2.rectangle(img, (cx - r + 1, cy - 4),
                          (cx + r - 3, cy + 3), col, -1)
            cv2.rectangle(img, (cx - r + 1, cy + 5),
                          (cx + r - 3, cy + 7), col, -1)
        elif kind == "eraser":
            cv2.rectangle(img, (cx - r + 1, cy - 5),
                          (cx + r - 3, cy + 6), col, 1)
            cv2.line(img, (cx - r + 1, cy + 6),
                     (cx + r - 3, cy + 6), col, 1)
        elif kind == "fill":
            pts = np.array([[cx - 6, cy - 6],
                            [cx + 6, cy],
                            [cx - 6, cy + 6]], np.int32)
            cv2.fillPoly(img, [pts], col)
            cv2.circle(img, (cx + 5, cy + 6), 2, col, -1)
        elif kind == "snap":
            cv2.rectangle(img, (cx - 7, cy - 7), (cx + 7, cy + 7), col, 1)
            cv2.line(img, (cx, cy - 7), (cx, cy + 7), col, 1)
            cv2.line(img, (cx - 7, cy), (cx + 7, cy), col, 1)
        elif kind == "shape":
            cv2.circle(img, (cx - 3, cy), 5, col, 1)
            cv2.rectangle(img, (cx + 1, cy - 5), (cx + 9, cy + 5), col, 1)
        elif kind == "undo":
            cv2.ellipse(img, (cx, cy + 2), (9, 7), 0, 180, 340, col, 2)
            cv2.arrowedLine(img, (cx - 6, cy - 5),
                            (cx - 2, cy - 9), col, 2, tipLength=0.4)
        elif kind == "redo":
            cv2.ellipse(img, (cx, cy + 2), (9, 7), 0, 200, 360, col, 2)
            cv2.arrowedLine(img, (cx + 6, cy - 5),
                            (cx + 2, cy - 9), col, 2, tipLength=0.4)
        elif kind == "save":
            cv2.rectangle(img, (cx - 7, cy - 6), (cx + 7, cy + 7), col, 1)
            cv2.rectangle(img, (cx - 3, cy - 6), (cx + 3, cy - 1), col, -1)
            cv2.rectangle(img, (cx - 4, cy + 2), (cx + 4, cy + 7), col, 1)
        elif kind == "clear":
            cv2.line(img, (cx - 6, cy - 6), (cx + 6, cy + 6), col, 2)
            cv2.line(img, (cx + 6, cy - 6), (cx - 6, cy + 6), col, 2)

    # ---------------------------------------------------------------
    # Hit test
    # ---------------------------------------------------------------
    def hit_test(self, point):
        if point is None:
            return None
        px, py = point
        if self.open_panel:
            hit = self._hit_test_panel((px, py))
            if hit is not None:
                return hit
        for bid, (x, y, w, h) in self.rects.items():
            if x <= px <= x + w and y <= py <= y + h:
                return bid
        return None

    def _panel_origin(self, anchor_btn):
        ax, ay, aw, ah = self.rects[anchor_btn]
        return (ax, ay + ah + 6)

    def _panel_layout(self):
        layout = {}
        if self.open_panel == "colors":
            ox, oy = self._panel_origin("colors")
            cols, rows = 6, 2
            cell, gap = 34, 6
            for i in range(len(settings.PALETTE)):
                r = i // cols
                c = i % cols
                layout[f"color_{i}"] = (ox + c * (cell + gap),
                                        oy + r * (cell + gap),
                                        cell, cell)
            return layout, (ox - 8, oy - 8,
                            cols * (cell + gap) + 16,
                            rows * (cell + gap) + 16)
        if self.open_panel == "brush":
            ox, oy = self._panel_origin("brush")
            cw, ch, gap = 140, 36, 6
            for i in range(len(settings.BRUSH_SIZES)):
                layout[f"brush_{i}"] = (ox, oy + i * (ch + gap), cw, ch)
            return layout, (ox - 8, oy - 8, cw + 16,
                            len(settings.BRUSH_SIZES) * (ch + gap) + 16)
        if self.open_panel == "shapes":
            ox, oy = self._panel_origin("shape_lock")
            cw, ch, gap = 150, 30, 5
            for i, _ in enumerate(settings.SHAPES):
                layout[f"shape_{i}"] = (ox, oy + i * (ch + gap), cw, ch)
            return layout, (ox - 8, oy - 8, cw + 16,
                            len(settings.SHAPES) * (ch + gap) + 16)
        return {}, None

    def _hit_test_panel(self, point):
        layout, _ = self._panel_layout()
        px, py = point
        for key, (x, y, w, h) in layout.items():
            if x <= px <= x + w and y <= py <= y + h:
                return key
        return None

    # ---------------------------------------------------------------
    # Click
    # ---------------------------------------------------------------
    def click(self, target):
        if target is None:
            return None
        if target == "colors":
            self.open_panel = "colors"
            return "colors_open"
        if target == "brush":
            self.open_panel = "brush"
            return "brush_open"
        if target == "shape_lock":
            self.open_panel = "shapes"
            return "shapes_open"
        if target == "shape_mode":
            self.snap_mode = not self.snap_mode
            return "shape_mode_toggle"
        if target == "clear":
            return "clear"
        if target == "undo":
            return "undo"
        if target == "redo":
            return "redo"
        if target == "save":
            return "save"
        if target == "pen":
            return "tool_pen"
        if target == "highlight":
            return "tool_highlight"
        if target == "eraser":
            return "tool_eraser"
        if target == "fill":
            return "tool_fill"
        if target.startswith("color_"):
            idx = int(target.split("_")[1])
            self.selected_color_idx = idx
            self.open_panel = None
            return f"color_{idx}"
        if target.startswith("brush_"):
            idx = int(target.split("_")[1])
            self.selected_brush_idx = idx
            self.open_panel = None
            return f"brush_{idx}"
        if target.startswith("shape_"):
            idx = int(target.split("_")[1])
            self.selected_shape = settings.SHAPES[idx]
            self.open_panel = None
            return f"shape_{idx}"
        return None

    # ---------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------
    def render(self, frame, pointer_pt, hover_target):
        for bid, label, glyph in BUTTONS:
            rect = self.rects[bid]
            hovered = (hover_target == bid)
            active = False
            if bid == "shape_mode":
                active = self.snap_mode
            if bid in ("colors", "brush", "shape_lock"):
                active = (self.open_panel ==
                          {"colors": "colors",
                           "brush": "brush",
                           "shape_lock": "shapes"}[bid])
            self._draw_button(frame, rect, label, glyph, hovered, active)

        # color swatch at right end
        last = BUTTONS[-1][0]
        lx, ly, lw, lh = self.rects[last]
        sx = lx + lw + 16
        sy = ly + lh // 2
        cv2.circle(frame, (sx, sy), 11,
                   settings.PALETTE[self.selected_color_idx], -1)
        cv2.circle(frame, (sx, sy), 11, (220, 220, 220), 1)

        if self.open_panel:
            self._draw_panel(frame, hover_target)

        if pointer_pt is not None:
            px, py = int(pointer_pt[0]), int(pointer_pt[1])
            cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

    def _draw_panel(self, frame, hover_target):
        layout, panel_rect = self._panel_layout()
        if not layout:
            return
        px, py, pw, ph = panel_rect
        self._rounded(frame, (px, py), (px + pw, py + ph),
                      (30, 30, 30), radius=10, alpha=0.93)
        self._rounded(frame, (px, py), (px + pw, py + ph),
                      (110, 110, 110), radius=10, thickness=1)

        if self.open_panel == "colors":
            for i, color in enumerate(settings.PALETTE):
                key = f"color_{i}"
                x, y, w, h = layout[key]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
                border = (255, 255, 255) if i == self.selected_color_idx else (90, 90, 90)
                cv2.rectangle(frame, (x, y), (x + w, y + h), border, 2)
                if hover_target == key:
                    cv2.rectangle(frame, (x - 2, y - 2),
                                  (x + w + 2, y + h + 2), (0, 255, 255), 2)

        if self.open_panel == "brush":
            for i, size in enumerate(settings.BRUSH_SIZES):
                key = f"brush_{i}"
                x, y, w, h = layout[key]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)
                cv2.putText(frame, settings.BRUSH_LABELS[i],
                            (x + 10, y + h // 2 + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
                cv2.circle(frame, (x + w - 22, y + h // 2),
                           max(2, size // 3), (230, 230, 230), -1)
                if i == self.selected_brush_idx:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 255), 2)
                if hover_target == key:
                    cv2.rectangle(frame, (x - 2, y - 2),
                                  (x + w + 2, y + h + 2), (0, 255, 255), 2)

        if self.open_panel == "shapes":
            for i, name in enumerate(settings.SHAPES):
                key = f"shape_{i}"
                x, y, w, h = layout[key]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)
                cv2.putText(frame, name, (x + 10, y + h // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
                if name == self.selected_shape:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 255), 2)
                if hover_target == key:
                    cv2.rectangle(frame, (x - 2, y - 2),
                                  (x + w + 2, y + h + 2), (0, 255, 255), 2)
