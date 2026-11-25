import time
import cv2


class Overlay:
    def __init__(self):
        self.toasts = []   # list of (text, expiry_time, color)

    # ------------- HUD -------------
    @staticmethod
    def draw_hud(frame, gesture, scale, tool_name, fps):
        cv2.putText(frame, f"Gesture: {gesture}", (300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Scale: {scale:.2f}", (300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Tool: {tool_name}", (300, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS {fps:4.1f}", (w - 140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # ------------- Cursor -------------
    @staticmethod
    def draw_cursor(frame, point, color=(0, 0, 255)):
        if point is None:
            return
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, color, 2)

    # ------------- Legend -------------
    @staticmethod
    def draw_legend(frame):
        lines = [
            "Right index only : Draw",
            "Left index only : Toolbar pointer",
            "Right thumb-index pinch : Move",
            "OK over toolbar : Select tool",
            "Left OK off toolbar : Snap shape",
            "Both hands index-up, others down : Scale",
            "Right open palm : Clear | C/Z/Y/S keys",
        ]
        h = frame.shape[0]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (300, h - 20 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ------------- Toast -------------
    def toast(self, text, color=(0, 255, 180), duration=1.5):
        self.toasts.append((text, time.time() + duration, color))

    def render_toasts(self, frame):
        now = time.time()
        self.toasts = [t for t in self.toasts if t[1] > now]
        h, w = frame.shape[:2]
        y = h - 60
        for i, (text, expiry, color) in enumerate(reversed(self.toasts)):
            alpha = min(1.0, (expiry - now) / 0.5)
            color_a = tuple(int(c * alpha) for c in color)
            pos = (w // 2 - 100, y - i * 30)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color_a, 2)
