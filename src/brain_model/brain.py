"""Assembles the two hemispheres, the corpus callosum and (optionally) a lesion into one model."""
import numpy as np

from .connectivity import CorpusCallosum
from .hemispheres import LeftHemisphere, RightHemisphere
from .lesion import Lesion


class Brain:
    def __init__(self, config: dict, with_lesion: bool = True):
        rng = np.random.default_rng(config["seed"])
        n = config["n_regions"]
        dyn = config["dynamics"]
        self.left = LeftHemisphere(n, config["left_hemisphere"], dyn, rng)
        self.right = RightHemisphere(n, config["right_hemisphere"], dyn, rng)
        cc = config["corpus_callosum"]
        self.callosum = CorpusCallosum(cc["strength"], n, cc.get("plasticity_rate", 0.0),
                                       cc.get("plasticity_gain", 0.0), cc.get("takeover_gain", 0.0))
        self.lesion = None
        if with_lesion and config.get("lesion"):
            target_name = config["lesion"]["hemisphere"]
            if target_name not in ("left", "right"):
                raise ValueError("lesion.hemisphere must be 'left' or 'right'")
            pair = (self.left, self.right) if target_name == "left" else (self.right, self.left)
            self.lesion = Lesion(config["lesion"], *pair)

    def step(self, t: int, noise_left: np.ndarray, noise_right: np.ndarray) -> dict:
        """One time step: exchange signals, update both hemispheres, then spread damage and adapt."""
        into_left, into_right = self.callosum.signals(self.left, self.right)
        task_left, task_right = self.left.task_input(t), self.right.task_input(t)
        cc = self.callosum
        self.left.step(t, into_left, noise_left, partner_task=cc.takeover_by_left * task_right)
        self.right.step(t, into_right, noise_right, partner_task=cc.takeover_by_right * task_left)
        if self.lesion:
            self.lesion.update(t)
        self.callosum.adapt(self.left, self.right)
        return {
            "left": self.left.x.copy(), "right": self.right.x.copy(),
            "left_damage": self.left.damage.copy(), "right_damage": self.right.damage.copy(),
            "into_left": into_left, "into_right": into_right,
            "left_task": task_left, "right_task": task_right,
        }
