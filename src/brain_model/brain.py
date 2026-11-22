"""Assembles the two hemispheres, the corpus callosum and (optionally) a lesion into one model."""
import numpy as np

from . import anatomy
from .connectivity import CorpusCallosum
from .hemispheres import LeftHemisphere, RightHemisphere
from .lesion import Lesion


class Brain:
    def __init__(self, config: dict, with_lesion: bool = True):
        rng = np.random.default_rng(config["seed"])
        n = config["n_regions"]
        dyn = config["dynamics"]
        anat = config.get("anatomy")
        self.left = LeftHemisphere(n, config["left_hemisphere"], dyn, rng, anat)
        self.right = RightHemisphere(n, config["right_hemisphere"], dyn, rng, anat)
        cc = config["corpus_callosum"]
        into_left, into_right = anatomy.callosal_wiring(anat, cc.get("wiring", "mirror"), n)
        self.callosum = CorpusCallosum(cc["strength"], n, cc.get("plasticity_rate", 0.0),
                                       cc.get("plasticity_gain", 0.0), cc.get("takeover_gain", 0.0),
                                       into_left, into_right)
        self.lesion = None
        if with_lesion and config.get("lesion"):
            target_name = config["lesion"]["hemisphere"]
            if target_name not in ("left", "right"):
                raise ValueError("lesion.hemisphere must be 'left' or 'right'")
            cc_ = self.callosum
            if target_name == "left":
                pair, routes = (self.left, self.right), (cc_.route_into_left, cc_.route_into_right)
            else:
                pair, routes = (self.right, self.left), (cc_.route_into_right, cc_.route_into_left)
            self.lesion = Lesion(config["lesion"], *pair, *routes)

    def step(self, t: int, noise_left: np.ndarray, noise_right: np.ndarray) -> dict:
        """One time step: exchange signals, update both hemispheres, then spread damage and adapt."""
        into_left, into_right = self.callosum.signals(self.left, self.right)
        parts_left, parts_right = self.left.task_parts(t), self.right.task_parts(t)
        task_left, task_right = sum(parts_left.values()), sum(parts_right.values())
        cc = self.callosum
        self.left.step(t, into_left, noise_left, partner_task=cc.takeover_by_left * task_right, task=task_left)
        self.right.step(t, into_right, noise_right, partner_task=cc.takeover_by_right * task_left, task=task_right)
        if self.lesion:
            self.lesion.update(t)
        self.callosum.adapt(self.left, self.right)
        out = {
            "left": self.left.x.copy(), "right": self.right.x.copy(),
            "left_damage": self.left.damage.copy(), "right_damage": self.right.damage.copy(),
            "into_left": into_left, "into_right": into_right,
            "left_task": task_left, "right_task": task_right,
        }
        for side, parts in (("left", parts_left), ("right", parts_right)):
            for name, value in parts.items():
                out[f"{side}_task:{name}"] = value
        return out
