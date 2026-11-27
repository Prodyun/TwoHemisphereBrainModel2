"""The corpus callosum: the bridge that lets the two hemispheres share signals.

`strength` sets the normal link strength (0 = cut, i.e. "split brain").
`wiring` sets who talks to whom across the midline:
  * mirror: area i only talks to the matching area i on the other side (simplest version)
  * real:   the measured left-right fibres from diffusion MRI (HCP). An area can send to several
            areas, and some areas have no detected direct link (tractography misses many of the
            callosal fibres that reach the outer cortex, a known limitation).
  * both:   real fibres plus a mirror link for every area.

Two kinds of plasticity (slow adaptation after damage), both controlled by `plasticity_rate`:
  1. Stronger links: links INTO damaged regions strengthen, up to strength * (1 + plasticity_gain).
  2. Takeover: the healthy hemisphere takes on more of the damaged side's task at the matching
     areas (up to 1 + takeover_gain times its normal share), then sends that work back across.
     This mirrors "contralesional recruitment" seen in patients, e.g. the right hemisphere
     helping with language after left-hemisphere stroke. The healthy side only learns about
     the damage through the callosum, so with strength 0 (split brain) there is no takeover.
"""
import numpy as np

from ..hemispheres import Hemisphere


class CorpusCallosum:
    def __init__(self, strength: float, n_regions: int, plasticity_rate: float = 0.0,
                 plasticity_gain: float = 0.0, takeover_gain: float = 0.0,
                 into_left: np.ndarray | None = None, into_right: np.ndarray | None = None,
                 inhibition: float = 0.0):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("corpus_callosum.strength must be between 0 and 1")
        if inhibition < 0:
            raise ValueError("corpus_callosum.inhibition must be >= 0")
        self.strength = strength
        # Interhemispheric inhibition (the "brake"): each side's activity turns down the matching
        # areas' own processing on the other side. 0 = off (the model before this was added).
        self.inhibition = inhibition
        self.plasticity_rate = plasticity_rate
        self.plasticity_gain = plasticity_gain
        self.takeover_gain = takeover_gain
        # Wiring matrices (receiver x sender); None = mirror (area i <-> area i).
        self.into_left_wiring = into_left
        self.into_right_wiring = into_right
        self.gain_into_left = np.full(n_regions, strength)
        self.gain_into_right = np.full(n_regions, strength)
        # Multipliers on how much of the OTHER side's task each hemisphere does (1 = normal share).
        self.takeover_by_left = np.ones(n_regions)
        self.takeover_by_right = np.ones(n_regions)
        # Extra strength on single fibres (receiver x sender), built up by paired stimulation
        # (Bridge Builder, connectivity/bridge_therapy.py). None = no change to the wiring.
        self.extra_into_left: np.ndarray | None = None
        self.extra_into_right: np.ndarray | None = None
        # Damage to the callosum itself (MS plaques, TBI axonal injury: connectivity/callosal_injury.py):
        # 1 = intact fibres into that receiver area, 0 = cut. Multiplies the signal arriving there.
        self.integrity_into_left = np.ones(n_regions)
        self.integrity_into_right = np.ones(n_regions)
        # Which RECEIVING areas feel the brake (1 = yes). All by default; `inhibition_jobs` limits it to
        # some jobs' areas (after a motor stroke the brake is strong onto movement areas: Murase 2004,
        # Spalletti 2017; in touch/visual cortex the callosum instead helps: Kokinovic & Medini 2018).
        self.brake_mask_left = np.ones(n_regions)
        self.brake_mask_right = np.ones(n_regions)

    def base_weights(self, into: str) -> np.ndarray:
        """The measured wiring as a full matrix (receiver x sender); mirror = identity."""
        w = self.into_left_wiring if into == "left" else self.into_right_wiring
        n = len(self.gain_into_left)
        return np.eye(n) if w is None else w

    @staticmethod
    def _route(wiring: np.ndarray | None, values: np.ndarray) -> np.ndarray:
        return values if wiring is None else wiring @ values

    def route_into_left(self, values_on_right: np.ndarray) -> np.ndarray:
        """Carry one value per right area across to the left areas, following the wiring."""
        return self._route(self.into_left_wiring, values_on_right)

    def route_into_right(self, values_on_left: np.ndarray) -> np.ndarray:
        return self._route(self.into_right_wiring, values_on_left)

    def signals(self, left: Hemisphere, right: Hemisphere, out_left: np.ndarray | float = 1.0,
                out_right: np.ndarray | float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Signals arriving at (left, right) this step, sent from the other side's current activity.

        A damaged sender transmits less (its output is scaled by its health). `out_left/right`
        scale what each sender transmits (e.g. brain stimulation of the sending areas)."""
        sent_right = out_right * right.health * right.x
        sent_left = out_left * left.health * left.x
        arriving_left = self.route_into_left(sent_right)
        arriving_right = self.route_into_right(sent_left)
        if self.extra_into_left is not None:
            arriving_left = arriving_left + self.extra_into_left @ sent_right
        if self.extra_into_right is not None:
            arriving_right = arriving_right + self.extra_into_right @ sent_left
        into_left = self.integrity_into_left * self.gain_into_left * arriving_left
        into_right = self.integrity_into_right * self.gain_into_right * arriving_right
        return into_left, into_right

    def brakes(self, left: Hemisphere, right: Hemisphere) -> tuple[np.ndarray | float, np.ndarray | float]:
        """How much of its own processing each side keeps under the other side's brake (1 = no brake).

        The brake grows with how active the sending areas are; it travels along the same callosal
        wiring and is cut with the callosum (split brain = no brake)."""
        if self.inhibition == 0 or self.strength == 0:
            return 1.0, 1.0
        from_right = self.integrity_into_left * self.route_into_left(right.health * np.abs(right.x))
        from_left = self.integrity_into_right * self.route_into_right(left.health * np.abs(left.x))
        k = self.inhibition * self.strength
        return (1.0 / (1.0 + k * self.brake_mask_left * from_right),
                1.0 / (1.0 + k * self.brake_mask_right * from_left))

    def adapt(self, left: Hemisphere, right: Hemisphere) -> None:
        """Move each link and takeover multiplier a little toward its target for the current damage."""
        rate = self.plasticity_rate
        connected = self.strength > 0
        for gain, takeover, receiver, partner in (
            (self.gain_into_left, self.takeover_by_left, left, right),
            (self.gain_into_right, self.takeover_by_right, right, left),
        ):
            gain += rate * (self.strength * (1 + self.plasticity_gain * receiver.damage) - gain)
            if connected:
                takeover += rate * (1 + self.takeover_gain * partner.damage - takeover)
