"""The HEALTHY side actively treats the damaged side: two "bridge therapies" (configs/care.yaml `bridge:`).

Both act on PAIRS: a healthy-side sender area A and a damaged-side receiver area B, e.g. R_6a -> L_6d.

1. Bridge Builder  (mode: pair)  -- personalised paired stimulation across the callosum (ccPAS).
   Real basis: cortico-cortical paired associative stimulation, a pulse at A followed a few ms later by a
   pulse at B, strengthens the A -> B pathway (Hebbian, spike-timing dependent); the reverse order weakens
   it. A sham-controlled Phase II stroke trial (44 patients) found larger arm gains with ccPAS than sham
   (Fugl-Meyer +6.3 vs +3.2; Brain Communications 2024). Nobody yet chooses the pair per patient.
   In the model: on every paired session step the fibre A -> B gets extra strength
        forward: E += rate * taper * (cap - E)          cap = max_gain x the average measured fibre
        reverse: E -= rate * taper * (E + base)          (down to no link at all)
   and the extra strength slowly fades (decay_per_day) when it is not refreshed.
   ASSUMPTION: one model step is one hour, so the few-millisecond gap between the two pulses is not
   simulated; only its ORDER (A then B, or B then A) is. Bridge Builder needs real fibres, so it does
   nothing in a split brain (the control).

2. Mirror Drive  (mode: mirror_drive)  -- a closed loop from the healthy side to the damaged side.
   Real basis: contralaterally controlled FES (the healthy hand's movement drives stimulation of the
   weak hand; Knutson 2016, Stroke) and brain-state-triggered TMS (feasible in stroke; Front Neurol 2024).
   In the model: during practice, the activity of healthy area A is read (EEG in the real world) and,
   when it carries the practised movement (A's activity has the same sign as the movement B should
   make, above `threshold`), damaged area B is stimulated with  gain x taper x activity(A).
   Controls with the same amount of stimulation:
        timing: open    -> fixed pulses on an even share (`duty`) of the practice steps
        timing: random  -> the same share of steps, chosen at random
   Mirror Drive reads the healthy side from OUTSIDE the head, so it is an artificial bridge: unlike
   Bridge Builder it still works when the callosum is cut or damaged (MS, TBI, split brain).

Both can be tapered off (`taper:`, see therapy.taper_factor): the Hand-off schedule.
"""
import numpy as np

from ..therapy import taper_factor


class BridgeTherapy:
    MODES = ("pair", "mirror_drive")

    def __init__(self, cfg: dict, callosum, damaged, healthy, steps_per_day: int, onset_step: int,
                 rng: np.random.Generator):
        modes = cfg.get("mode", [])
        self.modes = [modes] if isinstance(modes, str) else list(modes)
        unknown = set(self.modes) - set(self.MODES)
        if unknown:
            raise ValueError(f"bridge.mode must be among {self.MODES}, got {sorted(unknown)}")
        self.cc, self.damaged, self.healthy = callosum, damaged, healthy
        self.spd, self.onset = steps_per_day, onset_step
        self.start_day = cfg.get("start_day", 7)
        self.end_day = cfg.get("end_day")
        self.paired = cfg.get("paired", True)
        self.taper = cfg.get("taper") or {}
        self.rng = rng
        self.pairs = []
        for p in cfg.get("pairs", []):
            j, i = healthy.area_index(p["from"]), damaged.area_index(p["to"])
            self.pairs.append((j, i))
        if not self.pairs:
            raise ValueError("bridge.pairs needs at least one {from: <healthy area>, to: <damaged area>}")
        pc = cfg.get("pair", {})
        self.order = pc.get("order", "forward")
        if self.order not in ("forward", "reverse"):
            raise ValueError("bridge.pair.order must be 'forward' or 'reverse'")
        self.rate = float(pc.get("rate", 0.02))
        # Optional STDP timing (Bi & Poo 1998): `isi_ms` = time from the healthy-side pulse to the
        # damaged-side pulse. Positive (healthy first) strengthens, negative weakens, and the size of the
        # change follows the STDP window relative to the 8 ms gap used in ccPAS (Buch 2011), where `rate`
        # was set. Without isi_ms the plain forward/reverse rule above is used.
        self.isi_ms = pc.get("isi_ms")
        if self.isi_ms is not None:
            from ..calibration import stdp_kernel
            sp = {"a_plus": 1.0, "a_minus": float(pc.get("a_minus_over_a_plus", 0.35)),
                  "tau_plus": float(pc.get("tau_plus_ms", 16.8)), "tau_minus": float(pc.get("tau_minus_ms", 33.7))}
            k = stdp_kernel(float(self.isi_ms), **sp)
            ref = stdp_kernel(float(pc.get("reference_isi_ms", 8.0)), **sp)
            self.order = "forward" if k >= 0 else "reverse"
            self.rate *= abs(k) / ref
        self.decay = float(pc.get("decay_per_day", 0.01)) / steps_per_day
        into = damaged.name
        self.base = callosum.base_weights(into)
        positive = self.base[self.base > 0]
        self.cap = float(pc.get("max_gain", 1.0)) * (float(positive.mean()) if positive.size else 1.0)
        if "pair" in self.modes:
            n = len(damaged.x)
            E = np.zeros((n, n))
            if into == "left":
                callosum.extra_into_left = E
            else:
                callosum.extra_into_right = E
            self.E = E
        else:
            self.E = None
        md = cfg.get("mirror_drive", {})
        self.timing = md.get("timing", "closed")
        if self.timing not in ("closed", "open", "random"):
            raise ValueError("bridge.mirror_drive.timing must be 'closed', 'open' or 'random'")
        self.gain = float(md.get("gain", 0.5))
        self.threshold = float(md.get("threshold", 0.05))
        self.duty = float(md.get("duty", 0.5))
        self._amp = np.full(len(self.pairs), 0.3)     # running size of the healthy signal (for open/random)
        self.fired = 0
        self.practice_steps = 0
        self.last_fired = False

    # ---------- when ----------
    def day(self, t: int) -> float:
        return (t - self.onset) / self.spd

    def active(self, t: int, practising: bool) -> bool:
        if t < self.onset:
            return False
        d = self.day(t)
        if d < self.start_day or (self.end_day is not None and d >= self.end_day):
            return False
        return practising or not self.paired

    def strength(self, t: int) -> float:
        return taper_factor(self.taper, self.day(t))

    # ---------- Mirror Drive: extra input to the damaged areas this step ----------
    def drive(self, t: int, practising: bool, pattern: np.ndarray | None) -> np.ndarray | None:
        self.last_fired = False
        if "mirror_drive" not in self.modes or not self.active(t, practising) or pattern is None:
            return None
        f = self.strength(t)
        if f <= 0:
            return None
        self.practice_steps += 1
        out = np.zeros(len(self.damaged.x))
        for k, (j, i) in enumerate(self.pairs):
            a = float(self.healthy.x[j])
            self._amp[k] = 0.9 * self._amp[k] + 0.1 * abs(a)
            if self.timing == "closed":
                fire = a * np.sign(pattern[i]) > self.threshold
                amount = a
            elif self.timing == "open":
                fire = (self.practice_steps * self.duty) % 1.0 < self.duty - 1e-9 or self.duty >= 1.0
                amount = self._amp[k]
            else:
                fire = self.rng.random() < self.duty
                amount = self._amp[k]
            if fire:
                out[i] += self.gain * f * amount
                self.last_fired = True
        if self.last_fired:
            self.fired += 1
        return out

    # ---------- Bridge Builder: strengthen (or weaken) the chosen fibres ----------
    def update(self, t: int, practising: bool) -> None:
        if self.E is None:
            return
        if self.decay > 0:
            self.E *= 1.0 - self.decay
        if not self.active(t, practising) or self.cc.strength == 0:
            return
        r = self.rate * self.strength(t)
        if r <= 0:
            return
        for j, i in self.pairs:
            if self.order == "forward":
                self.E[i, j] += r * (self.cap - self.E[i, j])
            else:
                self.E[i, j] = max(-self.base[i, j], self.E[i, j] - r * (self.E[i, j] + self.base[i, j]))

    def pair_weights(self) -> np.ndarray:
        """The current strength of each chosen fibre (measured wiring + what pairing added)."""
        return np.array([self.base[i, j] + (self.E[i, j] if self.E is not None else 0.0) for j, i in self.pairs])

    def measured_duty(self) -> float:
        return self.fired / self.practice_steps if self.practice_steps else 0.0
