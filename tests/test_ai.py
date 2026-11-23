"""Tests for the AI Help Predictor pieces: local jobs, the help-pathway tracer, viewer data for
patient care, training-row building, and the model itself (on small synthetic data).

They use the tiny fake brain from test_hcp.py, so no downloads are needed.
"""
import numpy as np
import pytest

from brain_model.help_model import AreaFeatures, HelpPredictor, MODES, train
from brain_model.help_pathways import explain, trace
from brain_model.patient import deep_merge, ensure_job, patient_from_area
from brain_model.simulation import run_simulation
from brain_model.visualization.care_view import care_payload
from test_care import care_config, fake_file  # noqa: F401  (fixture)

CARE = {"local_job": {"neighbors": 2, "period": 40}}


# ---------- every area gets a job ----------

def test_area_in_a_job_keeps_its_job(fake_file):
    cfg = care_config(fake_file)
    assert ensure_job(cfg, "left", "L_44", CARE) == "language"
    assert "local_44" not in cfg["left_hemisphere"]["functions"]


def test_area_outside_any_job_gets_a_local_job(fake_file):
    cfg = care_config(fake_file)
    job = ensure_job(cfg, "left", "L_FFC", CARE)
    assert job == "local_FFC"
    areas = cfg["left_hemisphere"]["functions"][job]["areas"]
    assert areas[0] == "FFC" and len(areas) <= 3
    used = {a for name, f in cfg["left_hemisphere"]["functions"].items() if name != job for a in f["areas"]}
    assert not used.intersection(areas)          # never steals another job's areas


def test_patient_from_area_damages_the_area_and_its_neighbours(fake_file):
    p = patient_from_area("L_45", 0.8, "stroke", neighbors=2, anatomy_file=str(fake_file))
    assert p["damage"]["L_45"] == 0.8
    assert all(0 < v < 0.8 for k, v in p["damage"].items() if k != "L_45")
    assert all(k.startswith("L_") for k in p["damage"])


# ---------- the help-pathway tracer ----------

def test_trace_shares_sum_to_one_and_name_healthy_side_areas(fake_file):
    cfg = care_config(fake_file, corpus_callosum={"wiring": "both", "strength": 0.6})
    run = run_simulation(cfg)
    t = trace(cfg, run)
    assert t["healthy_hemisphere"] == "right" and t["total_help"] > 0
    shares = [h["share"] for h in t["helpers"]]
    assert all(h["area"].startswith("R_") for h in t["helpers"])
    assert sum(shares) == pytest.approx(1.0, abs=1e-3) or sum(shares) <= 1.0
    assert shares == sorted(shares, reverse=True)
    assert explain(t)


def test_split_brain_sends_no_help(fake_file):
    cfg = care_config(fake_file, corpus_callosum={"strength": 0.0})
    t = trace(cfg, run_simulation(cfg))
    assert t["total_help"] == 0 and t["helpers"] == []
    assert "No help" in explain(t)[0]


# ---------- viewer data for patient care ----------

def test_care_payload_has_every_plan_and_the_therapy_timeline(fake_file):
    cfg = care_config(fake_file, care={"goal": 0.8, "week_days": 7, "washout_steps": 1}, frame_stride=4,
                      analysis={"window": 40})
    cfg["care"] = {"goal": 0.8, "week_days": 2, "washout_steps": 1}
    plans = [("recommended", "AI plan: bci", deep_merge(cfg, {"therapy": {"mode": "bci"},
                                                             "stimulation": {"targets": [{"area": "R_44", "strength": 0.5}]}})),
             ("standard_care", "Standard care", cfg),
             ("split", "split", deep_merge(cfg, {"corpus_callosum": {"strength": 0.0}}))]
    p = care_payload(cfg, plans, prediction=None, helpers=[{"area": "R_44", "share": 0.6, "route": "direct fibre",
                                                             "reaches": ["L_44"]}])
    assert list(p["conditions"]) == ["recommended", "standard_care", "split"]
    n = p["meta"]["n_frames"]
    for c in p["conditions"].values():
        for key in ("practice", "stimulating", "bci_fired", "reps_physical", "reps_mental"):
            assert len(c[key]) == n
        assert "weeks_to_goal" in c["measures"] and c["weekly"]
    rec = p["conditions"]["recommended"]
    assert max(rec["practice"]) == 1 and max(rec["stimulating"]) == 1 and rec["targets"][0][3] == "R_44"
    assert p["conditions"]["split"]["strength"] == 0.0
    assert p["meta"]["care"]["helpers"][0]["area"] == "R_44"


# ---------- the model (small synthetic data with a known rule) ----------

def synthetic(n_areas=40, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"L_a{i}" for i in range(n_areas // 2)] + [f"R_a{i}" for i in range(n_areas // 2)]
    fibres = rng.integers(0, 15, n_areas).astype(float)
    table = np.column_stack([fibres, rng.normal(size=n_areas), rng.normal(size=n_areas)])
    feats = AreaFeatures(names, ["cc_fibres", "cc_strength", "homotopic_fc"], table,
                         {"cc_fibres": "number of direct fibres to the other hemisphere"})
    rows = []
    for i, a in enumerate(names):
        for sev in (0.6, 0.9):
            for mode in MODES:
                help_pts = 0.02 * fibres[i] + (0.03 if mode == "bci+vns" else 0) + rng.normal(0, 0.003)
                rows.append({"area": a, "severity": sev, "disorder": "stroke", "mode": mode, "help": help_pts,
                             "success": bool(fibres[i] > 6), "final_function": 0.6 + help_pts,
                             "never_impaired": False, "helpers": [{"area": "R_x", "share": 1.0}] if mode == "physical" else None})
    return rows, feats


def test_model_learns_a_known_rule_on_unseen_areas():
    rows, feats = synthetic()
    bundle = train(rows, feats, folds=4, max_iter=60)
    m = bundle["metrics"]
    assert m["auc_success"] > 0.9               # areas never seen in training
    assert m["r2_help"] > 0.8
    assert bundle["importance"][0][0] == "cc_fibres"


def test_predictor_ranks_modes_and_explains():
    rows, feats = synthetic()
    bundle = train(rows, feats, folds=4, max_iter=60)
    pred = HelpPredictor({k: v for k, v in bundle.items() if k != "oof"}, feats)
    many = int(np.argmax(feats.table[:, 0]))
    out = pred.predict(feats.names[many], 0.9, "stroke")
    assert out["best"]["chance"] > 0.5
    assert {m["mode"] for m in out["modes"]} == set(MODES)
    assert out["why"] and "fibres" in out["why"][0]
    with pytest.raises(ValueError):
        pred.predict("L_nowhere")
