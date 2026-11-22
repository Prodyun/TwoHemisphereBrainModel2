"""Rehabilitation (practice) and brain stimulation: the "help" the model can give a damaged hemisphere.

PRACTICE (`therapy:`) trains one job (e.g. "movement") of the damaged hemisphere in daily sessions.
During a session step the job's areas get extra input with the job's own time pattern, and the
step counts as `reps_per_step` repetitions. Areas used during practice heal (recovery.py).

One main mode (how the practice is done):
  physical            real movement: full input to the job's areas + sensory feedback from the
                      moving limb into the touch/position areas (3a, 3b, 1, 2). Counts PHYSICAL reps.
  imagery             imagining the movement: weaker input, weakest in the primary motor/touch areas
                      (imagery mainly uses premotor, SMA and parietal areas; M1 is weak or
                      inconsistent: Hardwick et al. 2018, Neurosci Biobehav Rev). No feedback.
                      Counts MENTAL reps (no physical effort).
  action_observation  watching the movement (video / mirror): like imagery, own strengths.
  bci                 brain-computer interface: imagery, and when the decoder sees the trained areas
                      become active (activity above `threshold`) it triggers feedback (e.g. muscle
                      stimulation, FES). Contingent feedback (it arrives exactly when the brain is
                      trying) gives a Hebbian bonus to healing; random-timed feedback does not
                      (Sci Rep 2024: +16.2 vs +4.8 Fugl-Meyer points). `decoder: healthy` reads the
                      healthy hemisphere instead (like the FDA-authorised IpsiHand device).
Add-ons (list them together with a main mode, e.g. [imagery, sleep_replay, vns]):
  sleep_replay        at the end of each practice day, a few "night" steps replay the practised
                      pattern weakly (targeted memory reactivation; only pilot studies in stroke).
  vns                 vagus nerve stimulation paired with each repetition multiplies healing during
                      practice (~2x in VNS-REHAB, Dawson et al. 2021 Lancet; 1.8-2.9x in rats). If
                      not paired (`paired: false`) it gives nothing, as in rats (Hays 2014).

Several main modes can be MIXED, e.g. [physical, imagery]: the practice hours of a session take
turns between them (physical hour, imagery hour, ...), so part of the repetitions are mental.

STIMULATION (`stimulation:`) changes how excitable chosen areas are, e.g. areas of the healthy
hemisphere (TMS / tDCS / focused ultrasound in the real world): excitatory rTMS makes an area
respond more strongly to its inputs, inhibitory rTMS less. In the model the area's own processing
is multiplied by (1 + strength): strength 0.5 = 50% more excitable, -0.3 = 30% calmer.
A more excitable healthy area relays the job's pattern more strongly across the corpus callosum.
`paired: true` = only during practice steps.
"""
from dataclasses import dataclass

import numpy as np

MAIN_MODES = ("physical", "imagery", "action_observation", "bci")
ADD_ONS = ("sleep_replay", "vns")


@dataclass
class PracticeStep:
    """What therapy does at one step."""
    input: np.ndarray            # extra task input for the trained hemisphere
    use_scale: np.ndarray        # per area: how much of its activity counts as "use" for healing
    boost: float                 # healing multiplier (VNS, contingent BCI feedback)
    kind: int                    # 0 none, 1 practice session, 2 sleep replay
    physical_reps: float
    mental_reps: float
    session_reps: float          # repetitions so far today (for saturation)
    pattern: np.ndarray | None = None  # the practised movement's pattern (for Hebbian "use")


def _modes(cfg: dict) -> tuple[list[str], set[str]]:
    raw = cfg.get("mode", "none")
    modes = [raw] if isinstance(raw, str) else list(raw)
    modes = [m for m in modes if m != "none"]
    unknown = set(modes) - set(MAIN_MODES) - set(ADD_ONS)
    if unknown:
        raise ValueError(f"unknown therapy mode(s) {sorted(unknown)}; choose from {MAIN_MODES + ADD_ONS}")
    main = [m for m in modes if m in MAIN_MODES]
    return main, set(modes) & set(ADD_ONS)


class Schedule:
    """Which steps are practice-session steps and which are night (sleep) steps."""

    def __init__(self, cfg: dict, steps_per_day: int, onset_step: int):
        self.spd = steps_per_day
        self.onset_step = onset_step
        self.start_day = cfg.get("start_day", 7)
        self.end_day = cfg.get("end_day")
        self.days_per_week = cfg.get("days_per_week", 5)
        self.session_steps = cfg.get("session_steps", 3)
        self.sleep_steps = cfg.get("sleep_replay", {}).get("steps", 2)
        if not 0 < self.session_steps < steps_per_day - self.sleep_steps:
            raise ValueError("therapy.session_steps must leave room in the day for the sleep steps")
        # Each day's session starts at a different hour (as in real life). The hours come from a fixed
        # seed, so every plan for a patient uses the same hours (a fair comparison). A fixed hour would
        # always skip the same phases of the job's rhythm when scoring, which biases the weekly score.
        self._starts: dict[int, int] = {}
        self._rng = np.random.default_rng(cfg.get("schedule_seed", 0))
        self._latest_start = max(0, steps_per_day - self.sleep_steps - self.session_steps - 2)

    def day(self, t: int) -> float:
        return (t - self.onset_step) / self.spd

    def is_practice_day(self, t: int) -> bool:
        day = int(np.floor(self.day(t)))
        if day < self.start_day or (self.end_day is not None and day >= self.end_day):
            return False
        return (day - self.start_day) % 7 < self.days_per_week

    def session_start(self, day: int) -> int:
        while len(self._starts) <= day:  # drawn in day order, so the hours never depend on the query order
            self._starts[len(self._starts)] = int(self._rng.integers(0, self._latest_start + 1))
        return self._starts[day]

    def kind(self, t: int) -> int:
        if t < self.onset_step or not self.is_practice_day(t):
            return 0
        step_in_day = (t - self.onset_step) % self.spd
        start = self.session_start(int(np.floor(self.day(t))))
        if start <= step_in_day < start + self.session_steps:
            return 1
        if step_in_day >= self.spd - self.sleep_steps:
            return 2
        return 0

    def step_in_session(self, t: int) -> int:
        day = int(np.floor(self.day(t)))
        return (t - self.onset_step) % self.spd - self.session_start(day)


class Therapy:
    def __init__(self, cfg: dict, trained, steps_per_day: int, onset_step: int, rng: np.random.Generator):
        self.mains, self.add_ons = _modes(cfg)
        self.main = self.mains[0] if self.mains else None
        self.trained = trained
        self.rng = rng
        self.schedule = Schedule(cfg, steps_per_day, onset_step)
        fname = cfg.get("function", "movement")
        funcs = {f.name: f for f in trained.functions}
        if fname not in funcs:
            raise ValueError(f"therapy.function '{fname}' is not a job of the {trained.name} hemisphere "
                             f"(choose from {sorted(funcs)})")
        self.function = fname
        self.regions = funcs[fname].regions
        n = len(trained.x)
        primary = {trained.area_index(a) for a in cfg.get("primary_areas", ["4", "3a", "3b", "1", "2"])
                   if self._known(trained, a)}
        self.is_primary = np.zeros(n, bool)
        self.is_primary[list(primary)] = True
        self.feedback_mask = np.zeros(n, bool)
        for a in cfg.get("feedback_areas", ["3a", "3b", "1", "2"]):
            if self._known(trained, a):
                self.feedback_mask[trained.area_index(a)] = True
        self.strength = cfg.get("strength", 1.0)
        self.reps_per_step = cfg.get("reps_per_step", 100)
        self.feedback = cfg.get("feedback", 0.3)
        self.imagery = cfg.get("imagery", {"primary": 0.35, "other": 0.75})
        self.observation = cfg.get("action_observation", {"primary": 0.3, "other": 0.6})
        bci = cfg.get("bci", {})
        self.bci_threshold = bci.get("threshold", 0.1)
        self.bci_contingent = bci.get("contingent", True)
        self.bci_bonus = bci.get("hebbian_bonus", 1.0)
        self.bci_random_rate = bci.get("random_rate", 0.5)
        self.bci_decoder = bci.get("decoder", "damaged")
        if self.bci_decoder not in ("damaged", "healthy"):
            raise ValueError("therapy.bci.decoder must be 'damaged' or 'healthy'")
        self.sleep_strength = cfg.get("sleep_replay", {}).get("strength", 0.1)
        vns = cfg.get("vns", {})
        self.vns_multiplier = vns.get("multiplier", 2.1)
        self.vns_paired = vns.get("paired", True)
        self._day, self._session_reps = None, 0.0

    @staticmethod
    def _known(hemi, area) -> bool:
        try:
            hemi.area_index(area)
            return True
        except (ValueError, TypeError):
            return False

    def _scale(self, main: str) -> np.ndarray:
        n = len(self.trained.x)
        scale = np.zeros(n)
        if main == "physical":
            scale[self.regions] = 1.0
        else:
            s = self.observation if main == "action_observation" else self.imagery
            scale[self.regions] = np.where(self.is_primary[self.regions], s["primary"], s["other"])
        return scale

    def step(self, t: int, pattern: np.ndarray, healthy_x: np.ndarray | None = None) -> PracticeStep:
        """`pattern` = the trained job's normal input at t (full-length vector)."""
        n = len(self.trained.x)
        zero = np.zeros(n)
        kind = self.schedule.kind(t)
        day = int(np.floor(self.schedule.day(t)))
        if day != self._day:
            self._day, self._session_reps = day, 0.0
        if kind == 2 and "sleep_replay" in self.add_ons:
            scale = np.zeros(n)
            scale[self.regions] = self.sleep_strength
            return PracticeStep(self.strength * scale * pattern, scale, 1.0, 2, 0.0, 0.0, self._session_reps, pattern)
        if kind != 1 or not self.mains:
            return PracticeStep(zero, zero, 1.0, 0, 0.0, 0.0, self._session_reps)
        main = self.mains[self.schedule.step_in_session(t) % len(self.mains)]  # mixed modes take turns
        scale = self._scale(main)
        inp = self.strength * scale * pattern
        boost = 1.0
        if main == "physical":
            inp = inp + self.feedback * np.abs(pattern) * self.feedback_mask
        elif main == "bci":
            watched = self.trained.x if self.bci_decoder == "damaged" or healthy_x is None else healthy_x
            detected = float(np.mean(np.abs(watched[self.regions]))) > self.bci_threshold
            fires = detected if self.bci_contingent else self.rng.random() < self.bci_random_rate
            if fires:  # e.g. muscle stimulation moves the hand: feedback reaches the touch areas
                inp = inp + self.feedback * np.abs(pattern) * self.feedback_mask
                if self.bci_contingent:
                    boost *= 1.0 + self.bci_bonus
        if "vns" in self.add_ons and self.vns_paired:
            boost *= self.vns_multiplier
        reps = self.reps_per_step
        self._session_reps += reps
        physical = reps if main == "physical" else 0.0
        return PracticeStep(inp, scale, boost, 1, physical, reps - physical, self._session_reps, pattern)


class Stimulation:
    """Excitability change at chosen areas (either hemisphere).

    focality_mm: a TMS coil does not hit one atlas area. Its electric field falls off over ~2-3 cm,
                 so neighbouring areas get a share exp(-d^2 / 2 sigma^2) of the strength (d = distance
                 between area centres). 0 = only the named area.
    callosal_output: a more excitable area also sends a stronger signal across the corpus callosum
                 (transcallosal facilitation, seen for the healthy-side dorsal premotor cortex after
                 stroke: Bestmann et al. 2010, J Neurosci)."""

    MM_PER_UNIT = 100.0  # model coordinates: 1 unit = 100 mm

    def __init__(self, cfg: dict, left, right, steps_per_day: int, onset_step: int):
        self.paired = cfg.get("paired", True)
        self.start_day = cfg.get("start_day", 7)
        self.end_day = cfg.get("end_day")
        self.callosal_output = cfg.get("callosal_output", True)
        sigma = cfg.get("focality_mm", 12.0) / self.MM_PER_UNIT
        self.spd, self.onset_step = steps_per_day, onset_step
        self.extra = {"left": np.zeros(len(left.x)), "right": np.zeros(len(right.x))}
        for target in cfg.get("targets", []):
            name = str(target["area"])
            hemi = left if name.startswith("L_") else right if name.startswith("R_") else None
            if hemi is None:
                raise ValueError(f"stimulation target {name} must start with L_ or R_")
            strength = float(target["strength"])
            if strength <= -1:
                raise ValueError("stimulation strength must be above -1 (-1 would switch the area off)")
            i = hemi.area_index(name)
            if sigma > 0:
                d = np.linalg.norm(hemi.centers - hemi.centers[i], axis=1)
                share = np.exp(-d ** 2 / (2 * sigma ** 2))
            else:
                share = np.zeros(len(hemi.x))
                share[i] = 1.0
            self.extra[hemi.name] += strength * share

    def output(self, side: str, t: int, practising: bool) -> np.ndarray | float:
        """Multiplier on what each area sends across the callosum (1 = unchanged)."""
        return self.gain(side, t, practising) if self.callosal_output else 1.0

    def active(self, t: int, practising: bool) -> bool:
        day = (t - self.onset_step) / self.spd
        if t < self.onset_step or day < self.start_day or (self.end_day is not None and day >= self.end_day):
            return False
        return practising or not self.paired

    def gain(self, side: str, t: int, practising: bool) -> np.ndarray | float:
        """Multiplier on each area's own processing right now (1 = unchanged)."""
        return np.clip(1.0 + self.extra[side], 0.0, None) if self.active(t, practising) else 1.0
