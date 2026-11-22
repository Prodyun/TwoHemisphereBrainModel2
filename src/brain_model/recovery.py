"""Healing: damaged areas recover a little when they are USED (use-dependent plasticity).

Each step, for every area of both hemispheres:

    damage -= window(days since onset) * (spontaneous + use_rate * boost * saturation * use) * (damage - floor)

  (spontaneous is set per DAY in the config, `spontaneous_per_day`, and spread over the day's steps;
  use_rate applies to each practice step.)

  * use          how strongly the area's activity moves TOGETHER with the practised movement
                 (Hebbian: max(0, activity x practised pattern)). The activity includes the signals
                 arriving from the healthy side through the corpus callosum, so healthy-side activity
                 that carries the practised movement counts; unrelated activity does not.
                 Plus `callosal_use_weight` x the matching input ARRIVING from the healthy side:
                 the callosal synapses that carry the practised movement into the damaged area are
                 strengthened as well (Hebbian pre x post). 0 = count only the area's own activity.
                 So help from the healthy side does not only cover for a damaged area: by keeping it
                 active during practice, it helps the area REPAIR. This is the project's main idea.
  * spontaneous  a small recovery that happens without therapy ("spontaneous recovery").
  * boost        > 1 when practice is paired with vagus nerve stimulation (VNS) or BCI feedback
                 that arrives at the right moment (see therapy.py).
  * saturation   1 / (1 + reps_this_session / saturation_reps): extra repetitions in one session
                 help less and less. Lang et al. 2016 (Ann Neurol) found no dose-response between
                 3,200 and 9,600 repetitions, so practice saturates.
  * window       the "golden period": how plastic the brain is at each time after the injury,
                 a table of (days, gain) points, interpolated in log-time. Default points follow the
                 Critical Period After Stroke Study (Dromerick et al. 2021, PNAS): extra therapy at
                 2-3 months helped most (+6.87 ARAT), at <=30 days less (+5.25), at >=6 months little
                 (+2.41, not significant). How fast the window closes has never been measured in
                 people, so the points are adjustable.
  * floor        permanent damage: floor_fraction x the worst damage the area ever had. Dead tissue
                 does not grow back; only the injured-but-alive part can recover.

Off unless `recovery.enabled: true`, so earlier results are unchanged.
"""
import numpy as np


class Recovery:
    def __init__(self, cfg: dict, n_regions: int, steps_per_day: float, onset_step: int):
        self.spontaneous = cfg.get("spontaneous_per_day", 0.0) / steps_per_day
        self.use_rate = cfg.get("use_rate", 0.0)
        self.floor_fraction = cfg.get("floor_fraction", 0.3)
        self.saturation_reps = cfg.get("saturation_reps", 300.0)
        self.callosal_use_weight = cfg.get("callosal_use_weight", 0.0)
        if not 0.0 <= self.floor_fraction <= 1.0:
            raise ValueError("recovery.floor_fraction must be between 0 and 1")
        points = np.array(cfg.get("window_points", [[0, 1.0]]), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or (np.diff(points[:, 0]) <= 0).any():
            raise ValueError("recovery.window_points must be [[days, gain], ...] with increasing days")
        self.window_days, self.window_gain = points[:, 0], points[:, 1]
        self.steps_per_day = steps_per_day
        self.onset_step = onset_step
        self.peak = {"left": np.zeros(n_regions), "right": np.zeros(n_regions)}

    def window(self, t: int) -> float:
        """Plasticity gain at step t (0 before the injury)."""
        if t < self.onset_step:
            return 0.0
        days = (t - self.onset_step) / self.steps_per_day
        # interpolate in log(1 + days): early days change fast, later months slowly
        return float(np.interp(np.log1p(days), np.log1p(self.window_days), self.window_gain))

    def floor(self, side: str) -> np.ndarray:
        return self.floor_fraction * self.peak[side]

    def update(self, t: int, hemi, use: np.ndarray | None, boost: float = 1.0,
               session_reps: float = 0.0) -> None:
        self.peak[hemi.name] = np.maximum(self.peak[hemi.name], hemi.damage)
        w = self.window(t)
        if w == 0.0:
            return
        rate = self.spontaneous
        if use is not None:
            saturation = 1.0 / (1.0 + session_reps / self.saturation_reps)
            rate = rate + self.use_rate * boost * saturation * use
        healable = np.maximum(hemi.damage - self.floor(hemi.name), 0.0)
        hemi.damage = hemi.damage - np.clip(w * rate, 0.0, 1.0) * healable
