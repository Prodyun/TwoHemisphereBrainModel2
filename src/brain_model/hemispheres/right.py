"""Right hemisphere: holistic / spatial processing (shapes, scenes, big-picture patterns)."""
import numpy as np

from .base import Hemisphere


class RightHemisphere(Hemisphere):
    name = "right"
    side = 1

    def task_input(self, t: int) -> np.ndarray:
        # Smooth waves that sweep across the whole hemisphere at once,
        # like taking in an entire picture rather than one piece at a time.
        y, z = self.centers[:, 1], self.centers[:, 2]
        return 0.5 * (np.sin(2 * np.pi * t / 60 + 3 * y) + np.cos(2 * np.pi * t / 90 + 4 * z))
