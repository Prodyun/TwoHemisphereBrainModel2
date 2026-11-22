"""Tests for the resting-brain fit and the fMRI reader (tiny fake data, no downloads needed)."""
import copy

import nibabel as nib
import numpy as np
import pytest

from brain_model import Brain
from brain_model.hcp.fmri import area_timeseries, subject_fc
from brain_model.simulation.fit import (block_scores, fit, rewire_bipartite, rewire_symmetric,
                                        sample_params)
from brain_model.simulation.rest import correlation_fc, rest_fc, rest_steps
from brain_model.utils import load_config


def test_rest_simulator_matches_the_full_brain():
    """The fast resting simulator must give exactly the Brain's numbers (no task, no lesion)."""
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg.update(n_regions=40)
    cfg["dynamics"]["stimulus"] = 0.0
    cfg["corpus_callosum"]["plasticity_rate"] = 0.0
    cfg["left_hemisphere"]["dynamics"] = {"gain": 0.7, "noise": 0.03}
    brain = Brain(cfg, with_lesion=False)
    noise = np.random.default_rng(0).normal(size=(2, 50, 40))
    p = {"leak_l": [0.2], "gain_l": [0.7], "noise_l": [0.03],
         "leak_r": [0.2], "gain_r": [0.5], "noise_r": [0.05], "strength": [0.5]}
    fast = list(rest_steps(brain.left.W, brain.right.W, None, None, p, noise))
    for t in range(50):
        out = brain.step(t, 0.03 * noise[0, t], 0.05 * noise[1, t])
        assert np.allclose(fast[t][0][0], out["left"]) and np.allclose(fast[t][1][0], out["right"])


def test_correlation_fc_basics():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(500, 1))
    ts = np.hstack([a, a + 0.1 * rng.normal(size=(500, 1)), rng.normal(size=(500, 1))])
    fc = correlation_fc(ts)
    assert np.allclose(fc, fc.T) and np.allclose(np.diag(fc), 0)
    assert fc[0, 1] > 1.5 and abs(fc[0, 2]) < 0.2


def test_block_scores_perfect_and_frozen():
    rng = np.random.default_rng(2)
    m = rng.normal(size=(8, 8))
    m = m + m.T
    assert block_scores(m, m)["score"] == pytest.approx(1.0)
    assert block_scores(np.zeros((8, 8)), m)["score"] == -1.0  # frozen activity ranks last


def test_rewiring_keeps_links_per_area_and_strengths():
    rng = np.random.default_rng(3)
    S = np.triu((rng.random((20, 20)) < 0.3) * rng.random((20, 20)), 1)
    S = S + S.T
    R = rewire_symmetric(S, rng)
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 0)
    assert (np.count_nonzero(R, axis=1) == np.count_nonzero(S, axis=1)).all()
    assert np.allclose(np.sort(R[R > 0]), np.sort(S[S > 0]))
    assert not np.allclose(R, S)
    B = (rng.random((10, 10)) < 0.3) * rng.random((10, 10))
    RB = rewire_bipartite(B, rng)
    assert (np.count_nonzero(RB, axis=1) == np.count_nonzero(B, axis=1)).all()
    assert (np.count_nonzero(RB, axis=0) == np.count_nonzero(B, axis=0)).all()


def test_identical_settings_are_tied():
    p = sample_params(5, np.random.default_rng(4), identical=True)
    for base in ("leak", "gain", "noise"):
        assert np.allclose(p[f"{base}_l"], p[f"{base}_r"])


def test_fit_recovers_a_known_setting_better_than_random():
    """Make 'real' FC from known settings, then check the fit scores well on it."""
    rng = np.random.default_rng(5)
    n = 10
    S = np.triu((rng.random((n, n)) < 0.5) * rng.random((n, n)), 1)
    W = S + S.T
    W = W / W.sum(axis=1, keepdims=True)
    noise = rng.normal(size=(2, 3000, n))
    truth = {"leak_l": [0.3], "gain_l": [0.9], "noise_l": [0.05],
             "leak_r": [0.3], "gain_r": [0.6], "noise_r": [0.05], "strength": [0.6]}
    emp = rest_fc(W, W, None, None, truth, noise)[0]
    res = fit(W, W, None, None, {"t": emp}, noise, rng, n_random=12, n_refine=8, batch=10)["t"]
    assert res["score"] > 0.8


def _fake_dtseries(path, n_time, left_vertices, right_vertices, data):
    bm = (nib.cifti2.BrainModelAxis.from_surface(left_vertices, 12, "CortexLeft")
          + nib.cifti2.BrainModelAxis.from_surface(right_vertices, 12, "CortexRight"))
    series = nib.cifti2.SeriesAxis(start=0, step=0.72, size=n_time)
    img = nib.cifti2.Cifti2Image(data.astype(np.float32), header=(series, bm))
    nib.save(img, str(path))


def test_fmri_is_averaged_per_area(tmp_path):
    # 12 surface vertices per side; the file only contains some of them (like the real medial wall)
    left_v, right_v = np.arange(0, 10), np.arange(2, 12)
    region = {"left": np.array([0, 0, 1, 1, 2, 2, -1, -1, 0, 1, 2, 2]),
              "right": np.array([-1, -1, 0, 0, 1, 1, 2, 2, 0, 1, 2, -1])}
    rng = np.random.default_rng(6)
    data = rng.normal(size=(30, 20))
    path = tmp_path / "run1.dtseries.nii"
    _fake_dtseries(path, 30, left_v, right_v, data)
    ts = area_timeseries(path, region, 3)
    assert ts.shape == (30, 6)
    # left area 0 = vertices 0, 1, 8 -> columns 0, 1, 8
    assert np.allclose(ts[:, 0], data[:, [0, 1, 8]].mean(axis=1), atol=1e-5)
    # right area 1 = vertices 4, 5, 9 -> columns 10 + (vertex - 2)
    assert np.allclose(ts[:, 4], data[:, [12, 13, 17]].mean(axis=1), atol=1e-5)
    fc = subject_fc([path, path], region, 3)
    assert fc.shape == (6, 6) and np.allclose(np.diag(fc), 0)
