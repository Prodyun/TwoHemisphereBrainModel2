"""A disorder (lesion) that starts in one region and spreads along the brain's wiring.

Spreading follows a network-spread model similar to those used for neurodegenerative
disease (e.g. Raj et al. 2012, Neuron): damage grows where there is already damage and
moves to connected regions.
"""
import numpy as np

from .hemispheres import Hemisphere


class Lesion:
    def __init__(self, cfg: dict, target: Hemisphere, other: Hemisphere):
        self.target, self.other = target, other
        self.start_step = cfg["start_step"]
        self.initial_damage = cfg["initial_damage"]
        self.spread_rate = cfg["spread_rate"]
        self.max_damage = cfg["max_damage"]
        self.cross_spread = cfg.get("cross_spread", 0.0)
        loc = np.asarray(cfg["location"], dtype=float)
        self.center_region = int(np.argmin(np.linalg.norm(target.centers - loc, axis=1)))

    def update(self, t: int) -> None:
        if t < self.start_step:
            return
        if t == self.start_step:
            self.target.damage[self.center_region] = self.initial_damage
        d_target, d_other = self.target.damage, self.other.damage  # use values from before this step
        for hemi, d, d_partner in ((self.target, d_target, d_other), (self.other, d_other, d_target)):
            growth = self.spread_rate * (1 - d) * (d + hemi.W @ d)
            leak = self.cross_spread * self.spread_rate * (1 - d) * d_partner
            hemi.damage = np.clip(d + growth + leak, 0.0, self.max_damage)
