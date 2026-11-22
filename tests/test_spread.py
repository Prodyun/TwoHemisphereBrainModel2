"""The network Fisher-Kolmogorov spreading model: damage travels outward along the wiring."""
import copy

import numpy as np
import pytest

from brain_model import Brain
from brain_model.utils import load_config


def fkpp_config(**lesion):
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg["lesion"].update({"model": "fisher_kpp", "transport": 0.01, "growth": 0.03, "start_step": 10, **lesion})
    return cfg


def run_damage(cfg, steps):
    brain = Brain(cfg)
    n = cfg["n_regions"]
    history = []
    for t in range(steps):
        brain.step(t, np.zeros(n), np.zeros(n))
        history.append(brain.left.damage.copy())
    return brain, np.array(history)


def hops_from(W, start):
    hop = np.full(len(W), -1)
    hop[start], frontier, k = 0, [start], 0
    while frontier:
        k += 1
        nxt = {j for i in frontier for j in np.flatnonzero(W[i] > 0) if hop[j] < 0}
        for j in nxt:
            hop[j] = k
        frontier = list(nxt)
    return hop


def test_no_damage_before_the_start_and_capped_after():
    cfg = fkpp_config()
    brain, d = run_damage(cfg, 400)
    assert np.all(d[:cfg["lesion"]["start_step"]] == 0)
    assert d.max() <= cfg["lesion"]["max_damage"] + 1e-12 and d.min() >= 0


def test_damage_arrives_in_order_of_wiring_distance():
    cfg = fkpp_config()
    brain, d = run_damage(cfg, 500)
    hop = hops_from(brain.left.W, brain.lesion.center_region)
    reached = d > 0.1
    arrival = np.where(reached.any(axis=0), reached.argmax(axis=0), np.inf)
    medians = [np.median(arrival[hop == h]) for h in (1, 2, 3) if (hop == h).any()]
    assert len(medians) >= 2
    assert all(a < b for a, b in zip(medians, medians[1:])), medians  # a front moving outward


def test_faster_transport_spreads_faster():
    _, slow = run_damage(fkpp_config(transport=0.005), 300)
    _, fast = run_damage(fkpp_config(transport=0.02), 300)
    assert fast[-1].mean() > slow[-1].mean()


def test_unknown_spread_model_rejected():
    with pytest.raises(ValueError):
        Brain(fkpp_config(model="teleport"))


def test_simple_model_is_still_the_default():
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    assert Brain(cfg).lesion.model == "simple"
