"""Assembles the two hemispheres and the corpus callosum into one model."""
import numpy as np

from .connectivity import CorpusCallosum
from .hemispheres import LeftHemisphere, RightHemisphere


class Brain:
    def __init__(self, config: dict):
        rng = np.random.default_rng(config["seed"])
        inp, hid = config["input_size"], config["hidden_size"]
        self.left = LeftHemisphere(inp, hid, config["left_hemisphere"]["learning_rate"], rng)
        self.right = RightHemisphere(inp, hid, config["right_hemisphere"]["learning_rate"], rng)
        self.callosum = CorpusCallosum(config["corpus_callosum"]["strength"])

    def step(self, stimulus: np.ndarray) -> dict:
        """One time step: both hemispheres see the stimulus, then exchange signals."""
        self.left.process(stimulus)
        self.right.process(stimulus)
        self.callosum.exchange(self.left, self.right)
        return {"left": self.left.state.copy(), "right": self.right.state.copy()}
