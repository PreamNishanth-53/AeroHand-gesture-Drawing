import cv2
import numpy as np
from config import settings


class ObjectTransform:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.center = [w // 2, h // 2]
        self.scale = 1.0
        self.grabbed = False
        self.grab_offset = (0, 0)

        # Two-hand continuous scale
        self.two_hand_active = False
        self.base_metric = None
        self.base_scale = 1.0
        self.two_hand_gain = 1.0

    # ---------------- RESET ----------------
    def reset(self):
        self.center = [self.w // 2, self.h // 2]
        self.scale = 1.0
        self.grabbed = False
        self.grab_offset = (0, 0)

        self.two_hand_active = False
        self.base_metric = None
        self.base_scale = 1.0

    # ---------------- MOVE ----------------
    def move_to(self, finger_pt):
        if not self.grabbed:
            self.grabbed = True
            self.grab_offset = (self.center[0] - finger_pt[0],
                                self.center[1] - finger_pt[1])
        self.center[0] = finger_pt[0] + self.grab_offset[0]
        self.center[1] = finger_pt[1] + self.grab_offset[1]

    def release(self):
        self.grabbed = False

    # ---------------- TWO-HAND SCALE ----------------
    def update_two_hand(self, valid, metric):
        """
        valid  : True when BOTH hands are in pinch pose this frame
        metric : hand_distance + pinch_spread_left + pinch_spread_right
                 (bigger metric -> bigger object)

        Continuous mapping:
            scale = base_scale * (metric / base_metric)

        The first time valid becomes True we capture the reference.
        The reference is only released when valid becomes False for
        a sustained moment (so a single-frame flicker does not lose it).
        """
        if valid:
            if not self.two_hand_active:
                # First valid frame -> anchor
                self.two_hand_active = True
                self.base_metric = max(metric, 1.0)
                self.base_scale = self.scale
            else:
                if self.base_metric is not None and self.base_metric > 0:
                    raw_ratio = metric / self.base_metric
                    ratio = 1.0 + (raw_ratio - 1.0) * self.two_hand_gain
                    new_scale = self.base_scale * ratio
                    self.scale = float(np.clip(new_scale,
                                               settings.MIN_SCALE,
                                               settings.MAX_SCALE))
            return True
        else:
            # Do not immediately drop the anchor; just stop scaling this
            # frame. If it stays invalid, main.py calls end_two_hand_scale.
            return self.two_hand_active

    def end_two_hand_scale(self):
        self.two_hand_active = False
        self.base_metric = None
        self.base_scale = 1.0

    # ---------------- RENDER ----------------
    def render(self, layer):
        M = cv2.getRotationMatrix2D((self.w // 2, self.h // 2), 0, self.scale)
        M[0, 2] += self.center[0] - self.w // 2
        M[1, 2] += self.center[1] - self.h // 2
        warped = cv2.warpAffine(layer, M, (self.w, self.h))
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
        return warped, mask
