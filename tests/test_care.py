"""Tests for patient care: damage maps, tumour growth, healing (recovery), practice (therapy),
brain stimulation, the interhemispheric brake, outcome measures and plan ranking.

They use the tiny fake brain from test_hcp.py (6 areas per hemisphere), so no downloads are needed.
"""
import copy

import numpy as np
import pytest

from brain_model import Brain
from brain_model.hcp.enigma import build_brain_file
from brain_model.patient import damaged_side, deep_merge
from brain_model.planning import parse_targets, rank_key
from brain_model.recovery import Recovery
from brain_model.simulation import run_simulation
from brain_model.therapy import Schedule
from test_hcp import fake_config, make_fake_enigma

SPD = 8


@pytest.fixture(scope="module")
def fake_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("care")
    make_fake_enigma(root / "raw")
    return build_brain_file(root / "raw", root / "hcp_brain.npz")


def care_config(path, **changes) -> dict:
    """Fake brain + a stroke damage map + recovery + physical practice of 'language' (areas 44, 45)."""
    cfg = fake_config(path)
    cfg.update(steps=SPD * 40, time={"steps_per_day": SPD})
    cfg["lesion"] = {**cfg["lesion"], "type": "stroke", "region": None, "start_step": SPD * 2,
                     "damage_map": {"L_44": 0.9, "L_45": 0.8}, "max_damage": 0.98}
    cfg["recovery"] = {"enabled": True, "spontaneous_per_day": 0.0, "use_rate": 0.05, "floor_fraction": 0.3,
                       "saturation_reps": 300, "callosal_use_weight": 1.0, "window_points": [[0, 1.0], [100, 1.0]]}
    cfg["therapy"] = {"mode": "physical", "function": "language", "start_day": 1, "days_per_week": 7,
                      "session_steps": 2, "reps_per_step": 150, "primary_areas": ["44"], "feedback_areas": ["45"],
                      "sleep_replay": {"strength": 0.1, "steps": 1}}
    return deep_merge(cfg, changes)


# ---------- nothing changes unless asked ----------

def test_care_parts_off_by_default_give_identical_runs(fake_file):
    cfg = fake_config(fake_file)
    off = deep_merge(cfg, {"therapy": {"mode": "none"}, "recovery": {"enabled": False},
                           "stimulation": {"targets": []}, "corpus_callosum": {"inhibition": 0.0}})
    a, b = run_simulation(cfg), run_simulation(off)
    assert np.array_equal(a["left"], b["left"]) and np.array_equal(a["left_damage"], b["left_damage"])
    assert "practice" not in a


# ---------- lesions from a patient's damage map; tumours ----------

def test_damage_map_can_hit_both_hemispheres(fake_file):
    cfg = care_config(fake_file, lesion={"damage_map": {"L_44": 0.9, "R_FFC": 0.4}})
    brain = Brain(cfg)
    brain.lesion._start()
    assert brain.left.damage[brain.left.area_index("44")] == 0.9
    assert brain.right.damage[brain.right.area_index("FFC")] == 0.4
    assert brain.left.region_names[brain.lesion.center_region] == "L_44"


def test_damage_map_rejects_bad_entries(fake_file):
    with pytest.raises(ValueError):
        Brain(care_config(fake_file, lesion={"damage_map": {"44": 0.5}}))
    with pytest.raises(ValueError):
        Brain(care_config(fake_file, lesion={"damage_map": {"L_44": 1.5}}))


def test_damaged_side_is_the_side_with_more_damage():
    assert damaged_side({"L_4": 0.2, "R_4": 0.9}) == "right"


def test_tumour_grows_gradually_and_presses_on_neighbours(fake_file):
    cfg = care_config(fake_file, lesion={"type": "tumor", "growth_steps": 80, "pressure": 0.3},
                      recovery={"enabled": False})
    cfg["lesion"]["damage_map"] = {"L_45": 0.2}   # replace (not merge) the default map
    run = run_simulation(cfg)
    i45 = Brain(cfg).left.area_index("45")
    centre = run["left_damage"][:, i45]
    start = cfg["lesion"]["start_step"]
    assert np.all(np.diff(centre[start:]) >= -1e-12)          # never shrinks
    assert centre[start + 40] < centre[start + 80]             # still growing halfway
    assert centre[-1] == pytest.approx(cfg["lesion"]["max_damage"])
    neighbours = np.delete(run["left_damage"][-1], i45)
    assert neighbours.max() == pytest.approx(0.3 * cfg["lesion"]["max_damage"])


# ---------- healing ----------

def test_window_is_zero_before_injury_and_follows_the_points():
    rec = Recovery({"window_points": [[0, 0.5], [10, 1.0]]}, 3, steps_per_day=4, onset_step=8)
    assert rec.window(7) == 0.0
    assert rec.window(8) == pytest.approx(0.5)
    assert rec.window(8 + 40) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Recovery({"window_points": [[5, 1.0], [2, 1.0]]}, 3, 4, 0)


def test_damage_never_heals_below_the_permanent_floor(fake_file):
    cfg = care_config(fake_file, recovery={"use_rate": 5.0, "spontaneous_per_day": 3.0})
    run = run_simulation(cfg)
    i44 = Brain(cfg).left.area_index("44")
    assert run["left_damage"][-1, i44] >= 0.3 * 0.9 - 1e-9
    assert run["left_damage"][-1, i44] < 0.9                   # it did heal


def test_practice_heals_more_than_no_practice(fake_file):
    with_p = run_simulation(care_config(fake_file))
    without = run_simulation(care_config(fake_file, therapy={"mode": "none"}))
    assert with_p["left_damage"][-1].sum() < without["left_damage"][-1].sum()


# ---------- practice ----------

def test_repetitions_are_counted_per_practice_step(fake_file):
    cfg = care_config(fake_file)
    run = run_simulation(cfg)
    practice_steps = int((np.asarray(run["practice"]) == 1).sum())
    assert practice_steps > 0
    assert run["reps_physical"][-1] == practice_steps * 150
    assert run["reps_mental"][-1] == 0


def test_imagery_counts_mental_reps_and_mixed_modes_take_turns(fake_file):
    img = run_simulation(care_config(fake_file, therapy={"mode": "imagery"}))
    assert img["reps_physical"][-1] == 0 and img["reps_mental"][-1] > 0
    mix = run_simulation(care_config(fake_file, therapy={"mode": ["physical", "imagery"]}))
    assert mix["reps_physical"][-1] == mix["reps_mental"][-1]


def test_schedule_keeps_rest_days_and_varies_the_session_hour():
    s = Schedule({"start_day": 0, "days_per_week": 5, "session_steps": 2, "sleep_replay": {"steps": 1}}, 24, 0)
    practice_days = {d for d in range(14) if any(s.kind(d * 24 + h) == 1 for h in range(24))}
    assert practice_days == {0, 1, 2, 3, 4, 7, 8, 9, 10, 11}
    starts = {s.session_start(d) for d in range(14)}
    assert len(starts) > 1


def test_vns_boosts_healing_only_when_paired(fake_file):
    base = run_simulation(care_config(fake_file))["left_damage"][-1].sum()
    paired = run_simulation(care_config(fake_file, therapy={"mode": ["physical", "vns"]}))["left_damage"][-1].sum()
    unpaired = run_simulation(care_config(fake_file, therapy={"mode": ["physical", "vns"],
                                                              "vns": {"paired": False}}))["left_damage"][-1].sum()
    assert paired < base
    assert unpaired == pytest.approx(base)


def test_bci_feedback_needs_the_brain_to_be_trying(fake_file):
    never = care_config(fake_file, therapy={"mode": "bci", "bci": {"threshold": 10.0, "contingent": True}})
    always = care_config(fake_file, therapy={"mode": "bci", "bci": {"threshold": -1.0, "contingent": True}})
    d_never = run_simulation(never)["left_damage"][-1].sum()
    d_always = run_simulation(always)["left_damage"][-1].sum()
    assert d_always < d_never   # contingent feedback (and its Hebbian bonus) only when activity is detected


def test_unknown_therapy_mode_or_job_rejected(fake_file):
    with pytest.raises(ValueError):
        Brain(care_config(fake_file, therapy={"mode": "telepathy"}))
    with pytest.raises(ValueError):
        Brain(care_config(fake_file, therapy={"function": "juggling"}))


# ---------- stimulation and the corpus callosum ----------

def test_stimulation_only_during_practice_when_paired(fake_file):
    cfg = care_config(fake_file, stimulation={"targets": [{"area": "R_44", "strength": 0.5}], "paired": True,
                                              "start_day": 1})
    run = run_simulation(cfg)
    stim, practice = np.asarray(run["stimulating"]), np.asarray(run["practice"]) == 1
    assert stim.any() and np.array_equal(stim, practice)


def test_stimulation_strength_of_minus_one_or_less_rejected(fake_file):
    with pytest.raises(ValueError):
        Brain(care_config(fake_file, stimulation={"targets": [{"area": "R_44", "strength": -1.0}]}))


def test_split_brain_gets_no_help_and_no_brake(fake_file):
    cfg = care_config(fake_file, corpus_callosum={"strength": 0.0, "inhibition": 2.0})
    run = run_simulation(cfg)
    assert np.allclose(run["into_left"], 0) and np.allclose(run["into_right"], 0)
    brain = Brain(cfg)
    assert brain.callosum.brakes(brain.left, brain.right) == (1.0, 1.0)


def test_brake_turns_down_the_other_side(fake_file):
    brain = Brain(care_config(fake_file, corpus_callosum={"inhibition": 2.0, "wiring": "mirror"}))
    brain.right.x[:] = 0.8
    left_keep, right_keep = brain.callosum.brakes(brain.left, brain.right)
    assert np.all(left_keep < 1.0) and np.all(right_keep == 1.0)  # the left is still silent


def test_healthy_side_help_speeds_healing(fake_file):
    """The project's core claim, in miniature: the same practice heals more with the callosum."""
    connected = run_simulation(care_config(fake_file, corpus_callosum={"wiring": "mirror", "strength": 0.6}))
    split = run_simulation(care_config(fake_file, corpus_callosum={"wiring": "mirror", "strength": 0.0}))
    assert connected["left_damage"][-1].sum() < split["left_damage"][-1].sum()


# ---------- ranking plans ----------

def test_plans_reaching_the_goal_rank_first_then_fewer_physical_reps():
    rows = [
        {"plan": "slow", "weeks_to_goal": None, "physical_reps_to_goal": None, "final_function": 0.79},
        {"plan": "phys", "weeks_to_goal": 8, "physical_reps_to_goal": 12000.0, "final_function": 0.85},
        {"plan": "bci", "weeks_to_goal": 10, "physical_reps_to_goal": 0.0, "final_function": 0.83},
    ]
    assert [r["plan"] for r in sorted(rows, key=rank_key)] == ["bci", "phys", "slow"]


def test_target_text_round_trips():
    assert parse_targets("R_4 +0.50; R_3b -0.30") == [{"area": "R_4", "strength": 0.5},
                                                      {"area": "R_3b", "strength": -0.3}]
    assert parse_targets("none") == []
