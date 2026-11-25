"""Tests for HEALTHY SIDE HELPS: Bridge Builder (paired stimulation strengthens one callosal fibre),
Mirror Drive (closed loop from the healthy side), the Hand-off taper and Help-or-Hinder decisions.
Uses the tiny fake brain from test_hcp.py."""
import numpy as np
import pytest

from brain_model import Brain
from brain_model.connectivity.bridge_therapy import BridgeTherapy
from brain_model.hcp.enigma import build_brain_file
from brain_model.healthy_side import better
from brain_model.patient import deep_merge
from brain_model.simulation import run_simulation
from brain_model.therapy import taper_factor
from test_care import SPD, care_config
from test_hcp import make_fake_enigma

PAIR = [{"from": "R_V1", "to": "L_V1"}]   # a fibre that exists in the fake brain
MIRROR = [{"from": "R_45", "to": "L_45"}]   # Mirror Drive needs a target in the practised job (language)


@pytest.fixture(scope="module")
def fake_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("bridge")
    make_fake_enigma(root / "raw")
    return build_brain_file(root / "raw", root / "hcp_brain.npz")


def bridge_cfg(path, **bridge):
    cfg = care_config(path)
    pairs = MIRROR if bridge.get("mode") == "mirror_drive" else PAIR
    return deep_merge(cfg, {"bridge": {"start_day": 1, "pairs": pairs, **bridge}})


def final_weight(cfg) -> float:
    return float(run_simulation(cfg)["bridge_weights"][-1][0])


def test_no_bridge_leaves_runs_unchanged(fake_file):
    cfg = care_config(fake_file)
    a = run_simulation(cfg)
    b = run_simulation(deep_merge(cfg, {"bridge": {"pairs": PAIR}}))   # no mode -> off
    assert np.array_equal(a["left"], b["left"]) and "bridge_weights" not in b


def test_pairing_forward_strengthens_and_reverse_weakens_the_fibre(fake_file):
    base = Brain(bridge_cfg(fake_file, mode="pair"), with_lesion=True).bridge.pair_weights()[0]
    fwd = final_weight(bridge_cfg(fake_file, mode="pair", pair={"order": "forward", "rate": 0.05}))
    rev = final_weight(bridge_cfg(fake_file, mode="pair", pair={"order": "reverse", "rate": 0.05}))
    assert fwd > base > rev >= 0.0


def test_pairing_needs_the_callosum(fake_file):
    cfg = deep_merge(bridge_cfg(fake_file, mode="pair"), {"corpus_callosum": {"strength": 0.0}})
    base = Brain(cfg, with_lesion=True).bridge.pair_weights()[0]
    assert final_weight(cfg) <= base + 1e-12   # split brain: no fibre to strengthen


def test_mirror_drive_fires_only_during_practice_and_controls_match_dose(fake_file):
    run = run_simulation(bridge_cfg(fake_file, mode="mirror_drive", mirror_drive={"timing": "closed"}))
    fired, practice = np.asarray(run["bridge_fired"]), np.asarray(run["practice"])
    assert fired.any() and not fired[practice != 1].any()
    opened = run_simulation(bridge_cfg(fake_file, mode="mirror_drive", mirror_drive={"timing": "open", "duty": 0.5}))
    f2 = np.asarray(opened["bridge_fired"])[np.asarray(opened["practice"]) == 1]
    assert abs(f2.mean() - 0.5) < 0.1


def test_mirror_drive_changes_the_damaged_side(fake_file):
    a = run_simulation(care_config(fake_file))
    b = run_simulation(bridge_cfg(fake_file, mode="mirror_drive", mirror_drive={"timing": "closed", "gain": 2.0}))
    assert not np.allclose(a["left"], b["left"])


def test_unknown_bridge_mode_is_rejected(fake_file):
    with pytest.raises(ValueError):
        Brain(bridge_cfg(fake_file, mode="telepathy"), with_lesion=True)


def test_taper_factor_shapes():
    assert taper_factor(None, 50) == 1.0
    assert taper_factor({"kind": "step", "to_day": 42}, 41) == 1.0
    assert taper_factor({"kind": "step", "to_day": 42}, 42) == 0.0
    lin = {"kind": "linear", "from_day": 28, "to_day": 70}
    assert taper_factor(lin, 20) == 1.0 and taper_factor(lin, 49) == pytest.approx(0.5) and taper_factor(lin, 70) == 0
    exp = {"kind": "exp", "from_day": 28, "to_day": 70}
    assert 0 < taper_factor(exp, 49) < taper_factor(lin, 49)


def test_stimulation_taper_switches_help_off(fake_file):
    cfg = deep_merge(care_config(fake_file), {"stimulation": {"start_day": 1, "paired": False,
                                                              "targets": [{"area": "R_44", "strength": 0.5}],
                                                              "taper": {"kind": "step", "to_day": 10}}})
    brain = Brain(cfg, with_lesion=True)
    onset = cfg["lesion"]["start_step"]
    assert np.max(brain.stimulation.gain("right", onset + 5 * SPD, True)) > 1.0
    assert brain.stimulation.gain("right", onset + 11 * SPD, True) is not None
    assert np.allclose(brain.stimulation.gain("right", onset + 11 * SPD, True), 1.0)


def test_better_needs_a_meaningful_difference():
    a = {"weeks_to_goal": 8, "final_function": 0.750}
    b = {"weeks_to_goal": 8, "final_function": 0.746}
    assert not better(a, b, 0.01)
    assert better({"weeks_to_goal": 7, "final_function": 0.70}, b, 0.01)
    assert better({"weeks_to_goal": None, "final_function": 0.9}, {"weeks_to_goal": None, "final_function": 0.8}, 0.01)
    assert not better({"weeks_to_goal": None, "final_function": 0.9}, b, 0.01)


def test_bridge_therapy_needs_pairs(fake_file):
    brain = Brain(care_config(fake_file), with_lesion=True)
    with pytest.raises(ValueError):
        BridgeTherapy({"mode": "pair", "pairs": []}, brain.callosum, brain.left, brain.right, SPD, 0,
                      np.random.default_rng(0))
