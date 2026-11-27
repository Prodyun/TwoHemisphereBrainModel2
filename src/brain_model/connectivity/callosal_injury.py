"""Damage to the BRIDGE itself: the corpus callosum's fibres (config `callosal_injury:`).

Each receiving area has an `integrity` (1 = its incoming callosal fibres are intact, 0 = cut). The signal
and the brake arriving there are multiplied by it (CorpusCallosum.integrity_into_left/right).

  kind: hit        traumatic brain injury (diffuse axonal injury): at the injury a share `fraction` of
                   the areas lose `severity` of their callosal input at once; afterwards a share
                   `recover_fraction` of the loss slowly returns over `recover_days`.
                   Basis: the corpus callosum is one of the most common sites of axonal injury after
                   TBI (Johnson et al. 2013, Brain).
  kind: relapsing  multiple sclerosis: relapses arrive at random (`relapses_per_year`). Each hits
                   `areas_per_relapse` random areas by `severity`; remyelination returns
                   `recover_fraction` of that loss over `recover_days`. On top, a slow loss
                   `progressive_per_year` everywhere. Basis: callosal atrophy grows with MS
                   disability (Ozturk et al. 2010; Evangelou et al. 2000).

All rates here are ASSUMPTIONS with sweep ranges (docs/factors.md): the papers show that the callosum is
damaged, not how fast in model units. Off unless configured, so existing runs are unchanged.
"""
import numpy as np


class CallosalInjury:
    KINDS = ("hit", "relapsing")

    def __init__(self, cfg: dict, callosum, steps_per_day: int, onset_step: int, rng: np.random.Generator):
        self.kind = cfg.get("kind", "hit")
        if self.kind not in self.KINDS:
            raise ValueError(f"callosal_injury.kind must be one of {self.KINDS}")
        self.cc, self.spd, self.onset, self.rng = callosum, steps_per_day, onset_step, rng
        self.n = len(callosum.gain_into_left)
        self.severity = float(cfg.get("severity", 0.5))
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("callosal_injury.severity must be between 0 and 1")
        self.fraction = float(cfg.get("fraction", 0.6))
        self.recover_fraction = float(cfg.get("recover_fraction", 0.3))
        self.recover_steps = max(1, int(cfg.get("recover_days", 60) * steps_per_day))
        self.relapse_p = float(cfg.get("relapses_per_year", 1.0)) / (365.0 * steps_per_day)
        self.areas_per_relapse = int(cfg.get("areas_per_relapse", 20))
        # Relapses become rarer with disease duration (Tremlett 2008): lambda(t) = lambda0 exp(-beta t),
        # t = years with the disease (starting at `years_with_disease`). beta = 0: constant rate.
        self.decline = float(cfg.get("relapse_decline_per_year", 0.0))
        self.years0 = float(cfg.get("years_with_disease", 0.0))
        self.progressive = float(cfg.get("progressive_per_year", 0.0)) / (365.0 * steps_per_day)
        # permanent loss and loss that can still heal, per side and area
        self.lost = {s: np.zeros(self.n) for s in ("left", "right")}
        self.healing = {s: np.zeros(self.n) for s in ("left", "right")}
        self.relapse_steps: list[int] = []

    def _hit(self, areas: dict[str, np.ndarray], amount: float) -> None:
        for side, idx in areas.items():
            have = 1.0 - self.lost[side][idx] - self.healing[side][idx]
            loss = amount * np.clip(have, 0.0, 1.0)
            self.healing[side][idx] += self.recover_fraction * loss
            self.lost[side][idx] += (1.0 - self.recover_fraction) * loss

    def _random_areas(self, k: int) -> dict[str, np.ndarray]:
        k = max(1, min(k, self.n))
        return {s: self.rng.choice(self.n, size=k, replace=False) for s in ("left", "right")}

    def update(self, t: int) -> None:
        if t < self.onset:
            return
        if t == self.onset and self.kind == "hit":
            self._hit(self._random_areas(int(round(self.fraction * self.n))), self.severity)
        p_now = self.relapse_p
        if self.decline:
            years = self.years0 + (t - self.onset) / (365.0 * self.spd)
            p_now = self.relapse_p * np.exp(-self.decline * years)
        if self.kind == "relapsing" and self.rng.random() < p_now:
            self.relapse_steps.append(t)
            self._hit(self._random_areas(self.areas_per_relapse), self.severity)
        for side in ("left", "right"):
            self.healing[side] *= 1.0 - 1.0 / self.recover_steps     # remyelination / repair
            if self.progressive:
                self.lost[side] = np.minimum(1.0, self.lost[side] + self.progressive * (1.0 - self.lost[side]))
        self.cc.integrity_into_left = np.clip(1.0 - self.lost["left"] - self.healing["left"], 0.0, 1.0)
        self.cc.integrity_into_right = np.clip(1.0 - self.lost["right"] - self.healing["right"], 0.0, 1.0)

    def mean_integrity(self) -> float:
        return float((self.cc.integrity_into_left.mean() + self.cc.integrity_into_right.mean()) / 2)
