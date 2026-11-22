"""A disorder (lesion) that starts in one area. Two kinds:

  * spreading: damage grows where there is already damage and moves along the brain's wiring.
      model "fisher_kpp" (default for the real brain): the network Fisher-Kolmogorov model used
        for prion-like spread of misfolded proteins (Weickenmeier et al. 2019, J Mech Phys Solids;
        Schafer et al. 2021). Each step:
            d += transport * (W_len @ d - d)  +  growth * d * (1 - d)
        Transport moves a little damage along each fibre (W_len = wiring divided by fibre length,
        with the distance between area centres standing in for fibre length); local growth then
        makes it take hold. Because transport is slow and growth fast, areas are reached one after
        another, by how far they are along the wiring from the start: a travelling front.
      model "simple": the earlier rule, growth = spread_rate * (1-d) * (d + W @ d).
  * stroke:    sudden damage to the starting area and its most strongly connected neighbours,
               which then does NOT spread.
  * tumor:     a slow-growing mass. Damage at the starting area climbs steadily from
               `initial_damage` to `max_damage` over `growth_steps`; its most strongly linked
               neighbours feel a weaker "pressure" (`pressure` x the centre's damage). No travelling
               front. Slow growth gives the healthy side time to adapt (Pasquini 2023; Ng 2023).

Where it starts: `region` (an area name such as "L_44", Broca's area; real brain only),
`location` (x, y, z: the nearest area is chosen), or `damage_map` ({area name: damage}, real brain
only): a patient's measured damage, on one or BOTH hemispheres (e.g. from a lesion scan). The most
damaged area of the target hemisphere becomes the centre.
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
        self.model = cfg.get("model", "simple")
        if self.model not in ("simple", "fisher_kpp"):
            raise ValueError("lesion.model must be 'simple' or 'fisher_kpp'")
        self.transport = cfg.get("transport", 0.0)
        self.growth = cfg.get("growth", 0.0)
        self.W_len = {h.name: self._length_weighted(h) for h in (target, other)}
        if self.kind not in ("spreading", "stroke", "tumor"):
            raise ValueError("lesion.type must be 'spreading', 'stroke' or 'tumor'")
        self.start_step = cfg["start_step"]
        self.initial_damage = cfg["initial_damage"]
        self.spread_rate = cfg.get("spread_rate", 0.0)
        self.max_damage = cfg["max_damage"]
        self.cross_spread = cfg.get("cross_spread", 0.0)
        self.stroke_neighbors = cfg.get("stroke_neighbors", 4)
        self.stroke_neighbor_damage = cfg.get("stroke_neighbor_damage", 0.7 * self.initial_damage)
        self.growth_steps = cfg.get("growth_steps", 600)
        self.pressure = cfg.get("pressure", 0.3)
        self.pressure_neighbors = cfg.get("pressure_neighbors", 4)
        self.damage_map = self._read_map(cfg.get("damage_map"))
        if self.damage_map:
            on_target = self.damage_map[target.name]
            if not on_target:
                raise ValueError(f"lesion.damage_map has no area in the {target.name} hemisphere")
            self.center_region = max(on_target, key=on_target.get)
        elif cfg.get("region") is not None:
            if target.region_names is None:
                raise ValueError("lesion.region needs the real brain (anatomy.source: hcp); use location")
            self.center_region = target.area_index(cfg["region"])
        else:
            loc = np.asarray(cfg["location"], dtype=float)
            self.center_region = int(np.argmin(np.linalg.norm(target.centers - loc, axis=1)))

    def _read_map(self, damage_map: dict | None) -> dict[str, dict[int, float]] | None:
        """{area name: damage} -> {hemisphere name: {area index: damage}}."""
        if not damage_map:
            return None
        out = {self.target.name: {}, self.other.name: {}}
        for name, value in damage_map.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"lesion.damage_map[{name}] must be between 0 and 1")
            if self.target.region_names is None:
                raise ValueError("lesion.damage_map needs the real brain (anatomy.source: hcp)")
            side = "left" if str(name).startswith("L_") else "right" if str(name).startswith("R_") else None
            if side is None:
                raise ValueError(f"damage_map area {name} must start with L_ or R_")
            hemi = self.target if self.target.name == side else self.other
            out[hemi.name][hemi.area_index(name)] = float(value)
        return out

    @staticmethod
    def _length_weighted(hemi: Hemisphere) -> np.ndarray:
        """Wiring divided by fibre length (centre-to-centre distance), rows scaled to sum to 1."""
        dist = np.linalg.norm(hemi.centers[:, None] - hemi.centers[None], axis=-1)
        W = np.divide(hemi.W, dist, out=np.zeros_like(hemi.W, dtype=float), where=dist > 0)
        s = W.sum(axis=1, keepdims=True)
        return np.divide(W, s, out=np.zeros_like(W), where=s > 0)

    def _start(self) -> None:
        if self.damage_map:
            for hemi in (self.target, self.other):
                for i, value in self.damage_map[hemi.name].items():
                    hemi.damage[i] = max(hemi.damage[i], min(value, self.max_damage))
            return
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
        if self.kind == "tumor":
            self._grow_tumor(t)
            return
        d_target, d_other = self.target.damage, self.other.damage  # use values from before this step
        for hemi, d, leak_in in ((self.target, d_target, self.into_target(d_other)),
                                 (self.other, d_other, self.into_other(d_target))):
            if self.model == "fisher_kpp":
                W = self.W_len[hemi.name]
                has_links = W.sum(axis=1) > 0
                change = self.transport * np.where(has_links, W @ d - d, 0.0) + self.growth * d * (1 - d)
                leak = self.cross_spread * self.transport * (1 - d) * leak_in
            else:
                change = self.spread_rate * (1 - d) * (d + hemi.W @ d)
                leak = self.cross_spread * self.spread_rate * (1 - d) * leak_in
            hemi.damage = np.clip(d + change + leak, 0.0, self.max_damage)

    def _grow_tumor(self, t: int) -> None:
        """Slow growth at the centre plus pressure on its strongest neighbours (no front)."""
        frac = min(1.0, (t - self.start_step) / max(1, self.growth_steps))
        d = self.target.damage
        start = self.damage_map[self.target.name][self.center_region] if self.damage_map else self.initial_damage
        centre = start + (self.max_damage - start) * frac
        d[self.center_region] = max(d[self.center_region], centre)
        row = self.target.W[self.center_region]
        nearest = [i for i in np.argsort(row)[::-1][: self.pressure_neighbors] if row[i] > 0]
        d[nearest] = np.maximum(d[nearest], self.pressure * centre)
