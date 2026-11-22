import copy

import numpy as np
import pytest

from brain_model import Brain
from brain_model import anatomy
from brain_model.connectivity import CorpusCallosum
from brain_model.simulation import function_score, run_experiment, run_simulation, summarize
from brain_model.utils import load_config
from brain_model.visualization import build_viewer

CONFIG = load_config("configs/default.yaml")
N = CONFIG["n_regions"]


def small(**changes) -> dict:
    """A shorter copy of the default config, with nested overrides like corpus_callosum={...}."""
    cfg = copy.deepcopy(CONFIG)
    cfg.update(steps=240, n_regions=60)
    cfg["lesion"]["start_step"] = 20
    cfg["lesion"]["spread_rate"] = 0.03
    cfg["analysis"]["window"] = 120
    for key, value in changes.items():
        cfg[key] = {**cfg[key], **value} if isinstance(value, dict) else value
    return cfg


def quiet_noise(n: int):
    return np.zeros(n), np.zeros(n)


# ---------- anatomy ----------

def test_hemispheres_are_mirror_images_on_opposite_sides():
    left, right = anatomy.region_centers(N, -1), anatomy.region_centers(N, 1)
    assert (left[:, 0] < 0).all() and (right[:, 0] > 0).all()
    assert np.allclose(left * [-1, 1, 1], right)


def test_connectivity_rows_sum_to_one_and_have_no_self_links():
    W = anatomy.structural_connectivity(anatomy.region_centers(N, -1), 8, 0.1, np.random.default_rng(0))
    assert np.allclose(W.sum(axis=1), 1)
    assert np.allclose(np.diag(W), 0)


def test_left_and_right_are_built_differently():
    brain = Brain(CONFIG)
    assert not np.allclose(brain.left.W, brain.right.W)
    assert not np.allclose(brain.left.task_input(10), brain.right.task_input(10))


def test_mesh_faces_point_to_valid_vertices():
    verts, faces = anatomy.hemisphere_mesh(-1, n_lat=8, n_lon=12)
    assert faces.min() == 0 and faces.max() < len(verts)


# ---------- corpus callosum ----------

def test_invalid_callosum_strength_rejected():
    with pytest.raises(ValueError):
        CorpusCallosum(1.5, N)


def test_split_brain_sends_no_signals():
    brain = Brain(small(corpus_callosum={"strength": 0.0}))
    brain.left.x[:] = brain.right.x[:] = 1.0
    into_left, into_right = brain.callosum.signals(brain.left, brain.right)
    assert not into_left.any() and not into_right.any()


def test_callosum_links_mirror_regions():
    brain = Brain(small())
    brain.right.x[:] = 0
    brain.right.x[5] = 1.0
    into_left, _ = brain.callosum.signals(brain.left, brain.right)
    assert into_left[5] > 0 and np.count_nonzero(into_left) == 1


def test_plasticity_strengthens_links_into_damaged_regions_only():
    brain = Brain(small())
    brain.left.damage[3] = 0.9
    for _ in range(200):
        brain.callosum.adapt(brain.left, brain.right)
    gains = brain.callosum.gain_into_left
    assert gains[3] > gains[4] == pytest.approx(brain.callosum.strength)
    assert brain.callosum.takeover_by_right[3] > 1 == pytest.approx(brain.callosum.takeover_by_right[4])


def test_no_takeover_in_split_brain():
    brain = Brain(small(corpus_callosum={"strength": 0.0}))
    brain.left.damage[:] = 0.9
    for _ in range(200):
        brain.callosum.adapt(brain.left, brain.right)
    assert np.allclose(brain.callosum.takeover_by_right, 1)


# ---------- lesion ----------

def test_damage_starts_on_time_and_spreads():
    cfg = small()
    brain = Brain(cfg)
    start = cfg["lesion"]["start_step"]
    for t in range(start):
        brain.step(t, *quiet_noise(cfg["n_regions"]))
    assert not brain.left.damage.any()
    brain.step(start, *quiet_noise(cfg["n_regions"]))
    first = np.count_nonzero(brain.left.damage)
    for t in range(start + 1, start + 60):
        brain.step(t, *quiet_noise(cfg["n_regions"]))
    assert np.count_nonzero(brain.left.damage) > first
    assert brain.left.damage.max() <= cfg["lesion"]["max_damage"] + 1e-9
    assert not brain.right.damage.any()  # cross_spread is 0 by default


def test_cross_spread_reaches_other_hemisphere():
    cfg = small(lesion={"cross_spread": 0.5})
    history = run_simulation(cfg)
    assert history["right_damage"][-1].max() > 0


def test_invalid_lesion_hemisphere_rejected():
    with pytest.raises(ValueError):
        Brain(small(lesion={"hemisphere": "middle"}))


def test_damage_weakens_a_region():
    brain = Brain(small())
    brain.left.x[:] = 0.5
    healthy = brain.left.step(0, np.zeros(60), np.zeros(60)).copy()
    brain.left.x[:] = 0.5
    brain.left.damage[:] = 0.9
    damaged = brain.left.step(0, np.zeros(60), np.zeros(60))
    assert np.abs(damaged).mean() < np.abs(healthy).mean()


# ---------- scoring and experiment ----------

def test_function_score_is_one_for_identical_runs_and_drops_for_silence():
    rng = np.random.default_rng(0)
    task = rng.normal(size=(100, 5))
    activity = 0.5 * task
    assert np.allclose(function_score(activity, activity, task, 50), 1)
    assert np.allclose(function_score(np.zeros_like(activity), activity, task, 50), 0)


def test_simulation_history_shapes():
    cfg = small()
    history = run_simulation(cfg)
    assert history["left"].shape == history["right_damage"].shape == (cfg["steps"], cfg["n_regions"])


def test_callosum_helps_damaged_hemisphere_in_default_experiment():
    """The main project result: with the callosum the damaged side keeps more function."""
    summary = summarize(run_experiment(CONFIG), CONFIG)
    assert summary["benefit_of_callosum"] > 0
    assert summary["split"]["healthy_side_score"] == pytest.approx(1.0)  # no link, so no effect


def test_viewer_file_is_built(tmp_path):
    cfg = small()
    results = run_experiment(cfg)
    path = build_viewer(cfg, results, summarize(results, cfg), tmp_path / "viewer.html")
    html = path.read_text(encoding="utf-8")
    assert "__BRAIN_DATA__" not in html and '"conditions"' in html
