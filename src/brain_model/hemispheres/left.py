"""Left hemisphere: sequential / analytical processing (language, logic, ordered steps)."""
import numpy as np

from .base import Hemisphere


class LeftHemisphere(Hemisphere):
    name = "left"
    side = -1
    N_GROUPS = 6       # the task is a sequence of 6 steps...
    STEP_LENGTH = 20   # ...each lasting 20 time steps

    def task_input(self, t: int) -> np.ndarray:
        # A pulse that moves front -> back through groups of regions, one group at a time,
        # like processing the words of a sentence in order.
        y = self.centers[:, 1]
        edges = np.quantile(-y, np.linspace(0, 1, self.N_GROUPS + 1)[1:-1])
        group = np.digitize(-y, edges)
        active = (t // self.STEP_LENGTH) % self.N_GROUPS
        return (group == active).astype(float)
