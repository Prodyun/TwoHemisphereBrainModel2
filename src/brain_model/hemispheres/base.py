"""Shared logic for both hemispheres: a network of brain regions with simple activity dynamics.

Each region has:
  * activity x  (-1..1): how active the region is right now
  * damage      (0..1):  0 = healthy, 1 = destroyed   (health = 1 - damage)

Each step a region's activity moves toward
    tanh( health * (input from its own neighbours + its task input)
          + input from its mirror region in the other hemisphere (corpus callosum)
          + noise )
The task input is mostly this hemisphere's own specialty, plus a weaker copy of the other
hemisphere's task (`shared_task`). The two sides are distinct but share some work.
Damage weakens a region's own processing. Signals arriving through the corpus callosum
come from the healthy side, so they can keep a damaged region partly working.
This is the model's key assumption; the experiment measures how much it helps.
"""
import numpy as np

from .. import anatomy


class Hemisphere:
    name = "base"
    side = 0  # -1 left, +1 right

    def __init__(self, n_regions: int, hemi_cfg: dict, dyn_cfg: dict, rng: np.random.Generator):
        self.centers = anatomy.region_centers(n_regions, self.side)
        self.W = anatomy.structural_connectivity(
            self.centers, hemi_cfg["neighbors"], hemi_cfg["long_range_fraction"], rng)
        self.leak = dyn_cfg["leak"]
        self.gain = dyn_cfg["gain"]
        self.stimulus_strength = dyn_cfg["stimulus"]
        self.shared_task = dyn_cfg.get("shared_task", 0.0)
        self.x = np.zeros(n_regions)
        self.damage = np.zeros(n_regions)

    @property
    def health(self) -> np.ndarray:
        return 1.0 - self.damage

    def task_input(self, t: int) -> np.ndarray:
        """The job this hemisphere is doing at time t. Left and right override this."""
        return np.zeros(len(self.x))

    def step(self, t: int, callosal_input: np.ndarray, noise: np.ndarray,
             partner_task: np.ndarray | None = None) -> np.ndarray:
        task = self.task_input(t)
        if partner_task is not None:
            task = task + self.shared_task * partner_task
        own = self.gain * self.W @ self.x + self.stimulus_strength * task
        drive = self.health * own + callosal_input + noise
        self.x = (1 - self.leak) * self.x + self.leak * np.tanh(drive)
        return self.x
