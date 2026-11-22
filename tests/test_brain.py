import numpy as np
import pytest

from brain_model import Brain
from brain_model.connectivity import CorpusCallosum
from brain_model.simulation import run_simulation
from brain_model.utils import load_config

CONFIG = load_config("configs/default.yaml")


def with_callosum(strength: float) -> dict:
    return {**CONFIG, "corpus_callosum": {"strength": strength}}


def test_step_returns_both_hemispheres():
    out = Brain(CONFIG).step(np.ones(CONFIG["input_size"]))
    assert out["left"].shape == out["right"].shape == (CONFIG["hidden_size"],)


def test_split_brain_hemispheres_stay_independent():
    brain = Brain(with_callosum(0.0))
    stim = np.ones(CONFIG["input_size"])
    out = brain.step(stim)
    assert np.allclose(out["left"], brain.left.process(stim))
    assert np.allclose(out["right"], brain.right.process(stim))


def test_full_connection_merges_hemispheres():
    out = Brain(with_callosum(1.0)).step(np.ones(CONFIG["input_size"]))
    assert np.allclose(out["left"], out["right"])


def test_partial_connection_keeps_hemispheres_distinct():
    out = Brain(with_callosum(0.5)).step(np.ones(CONFIG["input_size"]))
    assert not np.allclose(out["left"], out["right"])


def test_invalid_callosum_strength_rejected():
    with pytest.raises(ValueError):
        CorpusCallosum(1.5)


def test_simulation_history_shape():
    history = run_simulation({**CONFIG, "steps": 5})
    assert history["left"].shape == (5, CONFIG["hidden_size"])
