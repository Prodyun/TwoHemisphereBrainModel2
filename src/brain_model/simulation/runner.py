"""Runs the brain for N steps and collects the activity history."""
import numpy as np

from ..brain import Brain


def run_simulation(config: dict) -> dict:
    brain = Brain(config)
    rng = np.random.default_rng(config["seed"] + 1)
    history = {"left": [], "right": []}
    for _ in range(config["steps"]):
        stimulus = rng.normal(size=config["input_size"])
        out = brain.step(stimulus)
        history["left"].append(out["left"])
        history["right"].append(out["right"])
    return {k: np.array(v) for k, v in history.items()}
