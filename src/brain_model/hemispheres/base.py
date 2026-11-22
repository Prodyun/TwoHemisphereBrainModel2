"""Shared logic for both hemispheres: a network of brain regions with simple activity dynamics.

Each region has:
  * activity x  (-1..1): how active the region is right now
  * damage      (0..1):  0 = healthy, 1 = destroyed   (health = 1 - damage)

Each step a region's activity moves toward
    tanh( health * (input from its own neighbours + its task input)
          + input from the other hemisphere (corpus callosum)
          + noise )
This equation is the SAME for both hemispheres (a shared quality). What differs between left and
right is their wiring, their settings (leak, gain, noise can be set per hemisphere) and their
FUNCTIONS: each hemisphere does several jobs, and each job drives a set of brain areas with its
own time pattern (see FUNCTION PATTERNS below).

The task input is mostly this hemisphere's own functions, plus a weaker copy of the other
hemisphere's task at the matching areas (`shared_task`).
Damage weakens a region's own processing. Signals arriving through the corpus callosum
come from the healthy side, so they can keep a damaged region partly working.
This is the model's key assumption; the experiment measures how much it helps.
"""
from dataclasses import dataclass, field

import numpy as np

from .. import anatomy


# ---------- FUNCTION PATTERNS: how a job's input changes over time ----------
# Each returns the input (one value per area of the job) at time step t.

def sequence_groups(pos, groups=6):
    """Split areas into `groups` bands from front to back (used by the sequence pattern)."""
    groups = max(1, min(groups, len(pos)))
    y = pos[:, 1]
    edges = np.quantile(-y, np.linspace(0, 1, groups + 1)[1:-1])
    return np.digitize(-y, edges), groups


_SEQUENCE_BANDS: dict = {}


def _sequence(t, pos, step_length=20, groups=6):
    """A pulse that moves front -> back through groups of areas, one group at a time
    (like the words of a sentence, processed in order)."""
    key = (pos.tobytes(), groups)
    if key not in _SEQUENCE_BANDS:  # the bands never change, so work them out once
        _SEQUENCE_BANDS[key] = sequence_groups(pos, groups)
    group, n_groups = _SEQUENCE_BANDS[key]
    return (group == (t // step_length) % n_groups).astype(float)


def _wave(t, pos, period_a=60, period_b=90):
    """Smooth waves sweeping across all the job's areas at once (taking in a whole scene)."""
    y, z = pos[:, 1], pos[:, 2]
    return 0.5 * (np.sin(2 * np.pi * t / period_a + 3 * y) + np.cos(2 * np.pi * t / period_b + 4 * z))


def _ramp(t, pos, period=40, levels=4):
    """Input that climbs in steps then resets (counting / ordered steps)."""
    level = (t % period) // (period / levels) + 1
    return np.full(len(pos), level / levels)


def _bursts(t, pos, period=30, duration=8):
    """Short strong bursts (a face appears, then disappears)."""
    return np.full(len(pos), 1.0 if t % period < duration else 0.0)


def _sine(t, pos, period=50, phase=0.0):
    """A steady rhythm (a hand moving, a flickering scene, the tone of a voice)."""
    return np.full(len(pos), np.sin(2 * np.pi * t / period + phase))


PATTERNS = {"sequence": _sequence, "wave": _wave, "ramp": _ramp, "bursts": _bursts, "sine": _sine}


@dataclass
class Function:
    """One job of a hemisphere, e.g. 'language': which areas it uses and how its input looks over time."""
    name: str
    regions: np.ndarray
    pattern: str
    strength: float = 1.0
    params: dict = field(default_factory=dict)

    def input(self, t: int, centers: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n)
        out[self.regions] = self.strength * PATTERNS[self.pattern](t, centers[self.regions], **self.params)
        return out


class Hemisphere:
    name = "base"
    side = 0                # -1 left, +1 right
    default_pattern = None  # time pattern used when no functions are configured (stand-in mode)

    def __init__(self, n_regions: int, hemi_cfg: dict, dyn_cfg: dict, rng: np.random.Generator,
                 anatomy_cfg: dict | None = None):
        self.centers, self.W, self.region_names = anatomy.load_hemisphere(
            self.name, n_regions, hemi_cfg, anatomy_cfg, rng)
        dyn = {**dyn_cfg, **hemi_cfg.get("dynamics", {})}   # per-hemisphere settings override shared ones
        self.leak = dyn["leak"]
        self.gain = dyn["gain"]
        self.noise = dyn["noise"]
        self.stimulus_strength = dyn["stimulus"]
        self.shared_task = dyn.get("shared_task", 0.0)
        self.functions = self._build_functions(hemi_cfg, n_regions)
        self.x = np.zeros(n_regions)
        self.damage = np.zeros(n_regions)

    def _build_functions(self, hemi_cfg: dict, n: int) -> list[Function]:
        spec = hemi_cfg.get("functions")
        if not spec:  # stand-in mode: one job spread over every region
            return [Function(hemi_cfg.get("specialization", self.name), np.arange(n), self.default_pattern)]
        if self.region_names is None:
            raise ValueError("hemisphere functions name real brain areas, so they need anatomy.source: hcp")
        functions, used = [], set()
        for fname, f in spec.items():
            if f["pattern"] not in PATTERNS:
                raise ValueError(f"unknown pattern '{f['pattern']}' (choose from {sorted(PATTERNS)})")
            regions = [self.area_index(a) for a in f["areas"]]
            overlap = used.intersection(regions)
            if overlap:
                raise ValueError(f"{self.name} function '{fname}' reuses areas already used by another function")
            used.update(regions)
            functions.append(Function(fname, np.array(regions), f["pattern"], f.get("strength", 1.0),
                                      dict(f.get("params", {}))))
        return functions

    def area_index(self, name: str) -> int:
        prefix = "L_" if self.name == "left" else "R_"
        base = str(name).removesuffix("_ROI")
        if base[:2] in ("L_", "R_"):
            if base[:2] != prefix:
                raise ValueError(f"area {name} is not in the {self.name} hemisphere")
            base = base[2:]
        try:
            return self.region_names.index(prefix + base)
        except ValueError:
            raise ValueError(f"unknown {self.name} area: {name}") from None

    @property
    def health(self) -> np.ndarray:
        return 1.0 - self.damage

    def task_parts(self, t: int) -> dict[str, np.ndarray]:
        """Each function's input at time t (one full-length vector per function)."""
        n = len(self.x)
        return {f.name: f.input(t, self.centers, n) for f in self.functions}

    def task_input(self, t: int) -> np.ndarray:
        """Everything this hemisphere is doing at time t: the sum of its functions."""
        return sum(self.task_parts(t).values())

    def step(self, t: int, callosal_input: np.ndarray, noise: np.ndarray,
             partner_task: np.ndarray | None = None, task: np.ndarray | None = None) -> np.ndarray:
        if task is None:  # the Brain passes it in so it is only worked out once per step
            task = self.task_input(t)
        if partner_task is not None:
            task = task + self.shared_task * partner_task
        own = self.gain * self.W @ self.x + self.stimulus_strength * task
        drive = self.health * own + callosal_input + noise
        self.x = (1 - self.leak) * self.x + self.leak * np.tanh(drive)
        return self.x
