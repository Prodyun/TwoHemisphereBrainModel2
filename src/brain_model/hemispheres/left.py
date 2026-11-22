"""Left hemisphere: placeholder for language / sequential / analytical processing."""
import numpy as np

from .base import Hemisphere


class LeftHemisphere(Hemisphere):
    name = "left"

    def process(self, stimulus: np.ndarray) -> np.ndarray:
        # TODO: replace with a sequence/language model.
        # Placeholder: sharper, more selective activations.
        self.state = np.tanh(2.0 * (stimulus @ self.weights))
        return self.state
