"""Shared logic for both hemispheres."""
import numpy as np


class Hemisphere:
    """A minimal neural layer. The left/right subclasses specialise `process`."""

    name = "base"

    def __init__(self, input_size: int, hidden_size: int, learning_rate: float, rng: np.random.Generator):
        self.weights = rng.normal(0, 0.1, size=(input_size, hidden_size))
        self.learning_rate = learning_rate
        self.state = np.zeros(hidden_size)

    def process(self, stimulus: np.ndarray) -> np.ndarray:
        self.state = np.tanh(stimulus @ self.weights)
        return self.state

    def receive(self, signal: np.ndarray, strength: float) -> None:
        """Blend in a signal arriving from the other hemisphere."""
        self.state = (1 - strength) * self.state + strength * signal
