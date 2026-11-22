"""A disorder (lesion) that starts in one area. Two kinds:

  * spreading: damage grows where there is already damage and moves along the brain's wiring,
               a network-spread model like those used for neurodegenerative disease
               (e.g. Raj et al. 2012, Neuron).
  * stroke:    sudden damage to the starting area and its most strongly connected neighbours,
               which then does NOT spread.

Where it starts: `region` (an area name such as "L_44", Broca's area; real brain only) or
`location` (x, y, z: the nearest area is chosen).
`cross_spread` > 0 lets a spreading disorder leak to the other hemisphere along the callosal wiring.
"""
from typing import Callable

import numpy as np

from .hemispheres import Hemisphere

Route = Callable[[np.ndarray], np.ndarray]


def _same_area(values: np.ndarray) -> np.ndarray:
    return values


class Lesion:
    def __init__(self, cfg: dict, target: Hemisphere, other: Hemisphere,
                 into_target: Route = _same_area, into_other: Route = _same_area):
        self.target, self.other = target, other
        self.into_target, self.into_other = into_target, into_other
        self.kind = cfg.get("type", "spreading")
        if self.kind not in ("spreading", "stroke"):
            raise ValueError("lesion.type must be 'spreading' or 'stroke'")
        self.start_step = cfg["start_step"]
        self.initial_damage = cfg["initial_damage"]
        self.spread_rate = cfg["spread_rate"]
        self.max_damage = cfg["max_damage"]
        self.cross_spread = cfg.get("cross_spread", 0.0)
        self.stroke_neighbors = cfg.get("stroke_neighbors", 4)
        self.stroke_neighbor_damage = cfg.get("stroke_neighbor_damage", 0.7 * self.initial_damage)
        if cfg.get("region") is not None:
            if target.region_names is None:
                raise ValueError("lesion.region needs the real brain (anatomy.source: hcp); use location")
            self.center_region = target.area_index(cfg["region"])
        else:
            loc = np.asarray(cfg["location"], dtype=float)
            self.center_region = int(np.argmin(np.linalg.norm(target.centers - loc, axis=1)))

    def _start(self) -> None:
        d = self.target.damage
        d[self.center_region] = self.initial_damage
        if self.kind == "stroke":
            row = self.target.W[self.center_region]
            nearest = [i for i in np.argsort(row)[::-1][: self.stroke_neighbors] if row[i] > 0]
            d[nearest] = np.maximum(d[nearest], self.stroke_neighbor_damage)

    def update(self, t: int) -> None:
        if t < self.start_step:
            return
        if t == self.start_step:
            self._start()
        if self.kind == "stroke":
            return  # a stroke hits once and does not spread
        d_target, d_other = self.target.damage, self.other.damage  # use values from before this step
        for hemi, d, leak_in in ((self.target, d_target, self.into_target(d_other)),
                                 (self.other, d_other, self.into_other(d_target))):
            growth = self.spread_rate * (1 - d) * (d + hemi.W @ d)
            leak = self.cross_spread * self.spread_rate * (1 - d) * leak_in
            hemi.damage = np.clip(d + growth + leak, 0.0, self.max_damage)
