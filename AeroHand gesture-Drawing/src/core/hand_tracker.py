import cv2
import mediapipe as mp
from config import settings


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=settings.MAX_HANDS,   # will be 2 now
            min_detection_confidence=settings.DETECTION_CONF,
            min_tracking_confidence=settings.TRACKING_CONF,
        )

    def process(self, frame_bgr):
        """
        Returns (list_of_hands, raw_result).
        Each hand in the list: {
            "lm":  [(x, y), ... 21 points],
            "label": "Left" or "Right"  (from MediaPipe handedness)
        }
        List is sorted so Right hand always comes first (index 0).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return [], res

        h, w = frame_bgr.shape[:2]
        hands = []
        for hl, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            lm = [(int(p.x * w), int(p.y * h)) for p in hl.landmark]
            hands.append({"lm": lm, "label": label})

        # Sort so the "Right" hand (in camera frame) is always first.
        # MediaPipe labels mirror because we flipped the frame, so this is
        # just for stable ordering; either order works.
        hands.sort(key=lambda hnd: 0 if hnd["label"] == "Right" else 1)
        return hands, res

    def draw(self, frame, res):
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hl, self.mp_hands.HAND_CONNECTIONS
                )
