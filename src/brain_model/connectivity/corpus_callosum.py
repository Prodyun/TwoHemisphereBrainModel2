"""The corpus callosum: the bridge that lets the two hemispheres share signals.

Region i in one hemisphere is linked to its mirror region i in the other.
`strength` sets the normal link strength (0 = cut, i.e. "split brain").

Two kinds of plasticity (slow adaptation after damage), both controlled by `plasticity_rate`:
  1. Stronger links: links INTO damaged regions strengthen, up to strength * (1 + plasticity_gain).
  2. Takeover: the healthy hemisphere takes on more of the damaged side's task at the mirror
     regions (up to 1 + takeover_gain times its normal share), then sends that work back across.
     This mirrors "contralesional recruitment" seen in patients, e.g. the right hemisphere
     helping with language after left-hemisphere stroke. The healthy side only learns about
     the damage through the callosum, so with strength 0 (split brain) there is no takeover.
"""
import numpy as np

from ..hemispheres import Hemisphere


class CorpusCallosum:
    def __init__(self, strength: float, n_regions: int, plasticity_rate: float = 0.0,
                 plasticity_gain: float = 0.0, takeover_gain: float = 0.0):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("corpus_callosum.strength must be between 0 and 1")
        self.strength = strength
        self.plasticity_rate = plasticity_rate
        self.plasticity_gain = plasticity_gain
        self.takeover_gain = takeover_gain
        self.gain_into_left = np.full(n_regions, strength)
        self.gain_into_right = np.full(n_regions, strength)
        # Multipliers on how much of the OTHER side's task each hemisphere does (1 = normal share).
        self.takeover_by_left = np.ones(n_regions)
        self.takeover_by_right = np.ones(n_regions)

    def signals(self, left: Hemisphere, right: Hemisphere) -> tuple[np.ndarray, np.ndarray]:
        """Signals arriving at (left, right) this step, sent from the other side's current activity.

        A damaged sender transmits less (its output is scaled by its health)."""
        into_left = self.gain_into_left * right.health * right.x
        into_right = self.gain_into_right * left.health * left.x
        return into_left, into_right

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
