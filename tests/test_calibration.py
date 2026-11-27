"""The published formulas in calibration.py, and the model options that use them."""
import math

import numpy as np
import pytest
import yaml

from brain_model import calibration as cal
from brain_model.connectivity.corpus_callosum import CorpusCallosum
from brain_model.therapy import Schedule

TARGETS = yaml.safe_load(open("configs/calibration/targets.yaml", encoding="utf-8"))
GROUPS = TARGETS["healing"]["van_der_vliet_2020"]["groups"]


def test_recovery_curve_matches_the_paper_equation():
    # FM0 = 20, r = 0.7, tau = 2 weeks: after one time constant 63% of 0.7 x 46 is regained
    fm = cal.recovery_curve(20, [0, 2, 100], r=0.7, tau=2.0)
    assert fm[0] == pytest.approx(20)
    assert fm[1] == pytest.approx(20 + 0.7 * 46 * (1 - math.exp(-1)))
    assert fm[2] == pytest.approx(20 + 0.7 * 46, abs=1e-6)


def test_subgroup_weights_follow_the_starting_score():
    low, high = cal.subgroup_weights(3, GROUPS), cal.subgroup_weights(50, GROUPS)
    assert low.sum() == pytest.approx(1) and high.sum() == pytest.approx(1)
    assert np.argmax(low) == 0            # a near-zero arm is most likely the poor-recovery subgroup
    assert np.argmax(high) == 4           # a good arm is most likely subgroup 5 (r 0.93, tau 1.2 weeks)
    assert cal.expected_recovery(50, 26, GROUPS) > cal.expected_recovery(3, 26, GROUPS)


def test_rate_for_days_reaches_95_percent_on_time():
    rate = cal.rate_for_days(14, 24)
    assert rate == pytest.approx(0.00888, abs=1e-4)
    assert 1 - (1 - rate) ** (14 * 24) == pytest.approx(0.95)


def test_ihi_ratio_to_brake():
    k = cal.ihi_to_brake(0.5, 0.25)           # halves the response at activity 0.25
    assert 1 / (1 + k * 0.25) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cal.ihi_to_brake(1.5, 0.3)


def test_stdp_sign_follows_the_order():
    kw = {"a_plus": 1.0, "a_minus": 0.35, "tau_plus": 16.8, "tau_minus": 33.7}
    assert cal.stdp_kernel(8, **kw) > 0 > cal.stdp_kernel(-8, **kw)
    assert cal.stdp_kernel(8, **kw) > cal.stdp_kernel(40, **kw)     # shorter gap, stronger change


def test_epileptor_seizes_only_above_its_threshold():
    x0c = cal.epileptor_threshold()
    assert x0c == pytest.approx(-2.06, abs=0.01)
    assert cal.epileptor_seizes(x0c + 0.2)
    assert not cal.epileptor_seizes(x0c - 0.2)


def test_tumour_front_speed_and_network_rates():
    assert cal.pi_front_speed(0.13, 0.012) == pytest.approx(2 * math.sqrt(0.13 * 0.012))
    transport, growth = cal.pi_network_rates(0.13, 0.012, 30.0, 24)
    assert growth == pytest.approx(0.012 / 24)
    assert transport == pytest.approx(0.13 / 900 / 24)


def test_relapse_rate_declines():
    assert cal.relapse_rate(1.0, 0.0, 10) == 1.0
    assert cal.relapse_rate(1.0, 0.1, 10) == pytest.approx(math.exp(-1))


def test_extra_practice_adds_hours_only_in_its_window():
    s = Schedule({"start_day": 7, "days_per_week": 5, "session_steps": 2,
                  "extra": {"start_day": 75, "end_day": 103, "session_steps": 1}}, 24, 0)
    assert s.steps_today(10) == 2
    assert s.steps_today(77) == 3          # a weekday in the window (day 75 is a rest day)
    assert not s.in_extra(75)
    assert s.steps_today(110) == 2
    plain = Schedule({"start_day": 7, "days_per_week": 5, "session_steps": 2}, 24, 0)
    assert plain.steps_today(75) == 2


def test_brake_mask_limits_the_brake_to_chosen_areas():
    class H:
        def __init__(self, n):
            self.health = np.ones(n)
            self.x = np.ones(n)
    n = 4
    cc = CorpusCallosum(0.5, n, inhibition=2.0)
    cc.brake_mask_left[:] = [1, 1, 0, 0]
    left, right = cc.brakes(H(n), H(n))
    assert (left[:2] < 1).all() and (left[2:] == 1).all()
    assert (right < 1).all()


def test_brake_mask_built_from_jobs():
    from brain_model.brain import Brain
    from brain_model.utils import load_config
    cfg = load_config("configs/hcp.yaml")
    cfg["corpus_callosum"].update(inhibition=1.0, inhibition_jobs=["movement"])
    b = Brain(cfg, with_lesion=False)
    move = {f.name: f for f in b.left.functions}["movement"].regions
    assert b.callosum.brake_mask_left[move].all()
    assert b.callosum.brake_mask_left.sum() == len(move)


def test_practice_lowers_the_floor_only_when_switched_on():
    from brain_model.recovery import Recovery

    class H:
        name = "left"
        def __init__(self):
            self.damage = np.array([0.8, 0.8])

    def run(rate):
        rec, h = Recovery({"spontaneous_per_day": 24.0, "use_rate": 0.5, "floor_fraction": 0.5,
                           "floor_use_rate": rate}, 2, 24, 0), H()
        for t in range(2000):
            rec.update(t, h, np.array([1.0, 0.0]))
        return h.damage
    old, new = run(0.0), run(0.01)
    assert old == pytest.approx([0.4, 0.4])          # old rule: both stop at the floor (0.5 x 0.8)
    assert new[0] < 0.1 and new[1] == pytest.approx(0.4)   # only the practised area goes below it
