"""Right hemisphere: placeholder for spatial / holistic / pattern processing."""
import numpy as np

from .base import Hemisphere


class RightHemisphere(Hemisphere):
    name = "right"

    def process(self, stimulus: np.ndarray) -> np.ndarray:
        # TODO: replace with a spatial/pattern model.
        # Placeholder: smoother, more distributed activations.
        self.state = np.tanh(0.5 * (stimulus @ self.weights))
        return self.state
