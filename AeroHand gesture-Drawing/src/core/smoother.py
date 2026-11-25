class EMA:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.prev = None

    def update(self, point):
        if self.prev is None:
            self.prev = point
        else:
            self.prev = (
                self.alpha * point[0] + (1 - self.alpha) * self.prev[0],
                self.alpha * point[1] + (1 - self.alpha) * self.prev[1],
            )
        return self.prev

    def reset(self):
        self.prev = None


class LandmarkSmoother:
    """
    Smooths all 21 landmarks. Keeps separate state per hand label
    (\"Left\" / \"Right\") so the two hands don't corrupt each other.
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = {}   # label -> list of (x, y)

    def update(self, label, lm):
        if label not in self.prev:
            self.prev[label] = [(float(x), float(y)) for x, y in lm]
        else:
            new = []
            for (px, py), (x, y) in zip(self.prev[label], lm):
                nx = self.alpha * x + (1 - self.alpha) * px
                ny = self.alpha * y + (1 - self.alpha) * py
                new.append((nx, ny))
            self.prev[label] = new
        return [(int(x), int(y)) for x, y in self.prev[label]]

    def forget(self, label):
        self.prev.pop(label, None)

    def reset(self):
        self.prev.clear()


class GestureDebouncer:
    def __init__(self, required_frames=4):
        self.required = required_frames
        self.candidate = None
        self.count = 0
        self.current = None

    def update(self, gesture):
        if gesture == self.candidate:
            self.count += 1
        else:
            self.candidate = gesture
            self.count = 1

        if self.count >= self.required:
            self.current = self.candidate
        return self.current

    def reset(self):
        self.candidate = None
        self.count = 0
        self.current = None
