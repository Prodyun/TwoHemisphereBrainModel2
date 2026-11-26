"""Tests for the Clinic: the disease registry, the new engines (callosal injury, epilepsy), patient
intake, the spread-risk map and the Disease Builder. Engine tests use the tiny fake brain; tests that
need the real brain or the MNI atlas are skipped when those files are missing (they are never committed)."""
from pathlib import Path

import numpy as np
import pytest

from brain_model import Brain
from brain_model import diseases as D
from brain_model import intake
from brain_model.hcp.enigma import build_brain_file
from brain_model.patient import deep_merge
from brain_model.simulation import run_simulation
from brain_model.utils import load_config
from test_care import care_config
from test_hcp import make_fake_enigma

REAL = Path("data/processed/hcp_brain.npz")
needs_real = pytest.mark.skipif(not REAL.exists(), reason="needs the real brain (scripts/prepare_hcp.py)")
needs_atlas = pytest.mark.skipif(not intake.ATLAS.exists(), reason="needs the MNI Glasser atlas (scripts/fetch_area_data.py)")


@pytest.fixture(scope="module")
def fake_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("dis")
    make_fake_enigma(root / "raw")
    return build_brain_file(root / "raw", root / "hcp_brain.npz")


# ---------- the registry ----------

def test_every_profile_loads_with_plain_text_and_sources():
    profs = D.load_profiles()
    for name in ("stroke", "tumor", "alzheimers", "parkinsons", "als", "ms", "tbi", "epilepsy"):
        p = profs[name]
        assert p["plain"]["what"] and p["sources"], name
        assert len(p["plain"]["what"].split()) <= 14, f"{name}: keep the one-liner short"
    tabs = D.tabs()
    assert [t["name"] for t in tabs][:2] == ["stroke", "tumor"] and "spreading" not in [t["name"] for t in tabs]


def test_bad_profiles_are_rejected():
    with pytest.raises(ValueError):
        D.validate({"name": "x", "engine": {"type": "magic"}})
    with pytest.raises(ValueError):
        D.validate({"name": "x", "engine": {"type": "stroke"}, "seed": {"mode": "areas"}})


def test_placeholders_are_resolved():
    out = D.resolve({"targets": [{"area": "@mirror"}, {"area": "@epicentre"}]}, "L_6d", "R_")
    assert out == {"targets": [{"area": "R_6d"}, {"area": "L_6d"}]}


def test_disease_builder_saves_a_loadable_profile(tmp_path):
    path = D.save_custom({"name": "My Disease!", "engine": {"type": "stroke"}, "default_area": "L_4",
                          "severity": 0.6, "what": "a test"}, folder=tmp_path)
    assert path.name == "custom_my_disease.yaml"
    p = D.load_profiles(tmp_path)["custom_my_disease"]
    assert p["custom"] and p["tab"] == "My Disease!" and p["sources"] == []


# ---------- new engines on the fake brain ----------

def test_callosal_injury_lowers_the_signal_crossing(fake_file):
    cfg = care_config(fake_file)
    hit = deep_merge(cfg, {"callosal_injury": {"kind": "hit", "fraction": 1.0, "severity": 0.8, "recover_fraction": 0.0}})
    a, b = run_simulation(cfg), run_simulation(hit)
    assert b["callosal_integrity"][-1] < 0.3
    after = cfg["lesion"]["start_step"] + 2
    assert np.abs(b["into_left"][after:]).mean() < np.abs(a["into_left"][after:]).mean()


def test_relapses_and_repair(fake_file):
    cfg = deep_merge(care_config(fake_file), {"callosal_injury": {
        "kind": "relapsing", "relapses_per_year": 365 * 8, "areas_per_relapse": 2, "severity": 0.5,
        "recover_fraction": 1.0, "recover_days": 2}})
    run = run_simulation(cfg)
    integ = np.asarray(run["callosal_integrity"])
    assert integ.min() < 1.0               # relapses happen
    assert integ[-1] > integ.min()         # and full remyelination repairs some of it


def test_epilepsy_focus_has_seizures(fake_file):
    cfg = care_config(fake_file, lesion={"type": "epilepsy", "excitability": 1.8, "seizures_per_day": 4,
                                         "seizure_drive": 2.0})
    brain = Brain(cfg, with_lesion=True)
    run = run_simulation(cfg)
    assert np.asarray(run["seizure"]).sum() > 0
    brain.lesion._start()
    assert brain.left.excitability.max() == pytest.approx(1.8)


def test_default_hemisphere_settings_change_nothing(fake_file):
    brain = Brain(care_config(fake_file), with_lesion=True)
    assert (brain.left.excitability == 1).all() and (brain.left.seizure_drive == 0).all()
    assert (brain.callosum.integrity_into_left == 1).all()


# ---------- intake ----------

def test_table_and_suvr_intake():
    p = intake.from_table(intake.read_table("area,value\nL_6d,0.9\nL_4,0.4\n"))
    assert p["damage"] == {"L_6d": 0.9, "L_4": 0.4}
    t = intake.from_table([{"area": "L_EC", "value": 2.0}, {"area": "R_EC", "value": 1.1}], kind="suvr")
    assert "R_EC" not in t["damage"] and t["damage"]["L_EC"] > 0.8   # below the cut-off = no tau
    with pytest.raises(ValueError):
        intake.from_table([{"area": "EC", "value": 0.5}])


def test_symmetry_index_and_symptoms():
    si = intake.symmetry_index(12, 6)
    assert si["side"] == "left" and si["bsi"] == pytest.approx(1 / 3, abs=1e-3)
    assert intake.symmetry_index(10, 10.5)["side"] is None
    cfg = load_config("configs/hcp.yaml")
    p = intake.from_symptoms(["weak right hand or arm"], cfg)
    assert all(a.startswith("L_") for a in p["damage"]) and "L_4" in p["damage"]
    q = intake.from_symptoms(["weak right hand or arm", "trouble recognising faces"], cfg,
                             eeg={"delta_left": 5, "delta_right": 15})
    assert all(a.startswith("R_") for a in q["damage"])     # the EEG side decides


@needs_atlas
def test_lesion_mask_maps_onto_glasser_areas(tmp_path):
    import nibabel as nib
    at = nib.load(str(intake.ATLAS))
    ijk = np.indices(at.shape).reshape(3, -1)
    xyz = (at.affine @ np.r_[ijk, np.ones((1, ijk.shape[1]))])[:3]
    sphere = (np.linalg.norm(xyz.T - [-38, -22, 56], axis=1) < 12).reshape(at.shape).astype(np.uint8)
    path = tmp_path / "m1.nii.gz"
    nib.save(nib.Nifti1Image(sphere, at.affine), str(path))
    p = intake.from_mask(path)
    top = max(p["damage"], key=p["damage"].get)
    assert top in ("L_3b", "L_4", "L_3a") and all(a.startswith("L_") for a in p["damage"])


# ---------- real-brain checks ----------

@needs_real
@pytest.mark.parametrize("name", ["alzheimers", "parkinsons", "als", "ms", "tbi", "epilepsy"])
def test_each_disease_builds_a_patient(name):
    from brain_model.patient import patient_config
    p = D.get(name)
    cfg = patient_config(load_config("configs/hcp.yaml"), load_config("configs/care.yaml"), D.make_patient(p))
    plans = D.plan_list(cfg, p)
    assert any(k == "split" for k, _, _ in plans)
    if name == "als":
        dm = cfg["lesion"]["damage_map"]
        assert any(a.startswith("L_") for a in dm) and any(a.startswith("R_") for a in dm)
    if name == "alzheimers":
        assert cfg["therapy"]["function"] == "memory"


@needs_real
def test_spread_risk_grows_over_time():
    from brain_model.patient import patient_config
    from brain_model.spread_risk import risk
    p = D.get("alzheimers")
    cfg = patient_config(load_config("configs/hcp.yaml"), load_config("configs/care.yaml"), D.make_patient(p))
    r = risk(cfg, p, n_runs=4)
    days = [str(d) for d in r["days"]]
    for side in ("left", "right"):
        a, b = np.array(r["p"][days[0]][side]), np.array(r["p"][days[-1]][side])
        assert (b >= a).all()
    assert r["next"] and r["next"][0]["day"] > 0


# ---------- the explorer's tab cache (care_service) ----------

def _bare_service():
    from brain_model.care_service import CareService
    svc = object.__new__(CareService)   # no AI model or brain needed for the cache logic
    svc._tab_cache = {}
    return svc


def test_a_tabs_default_patient_is_served_from_the_cache():
    svc = _bare_service()
    key = svc._tab_key("L_6d", 0.9, "stroke", None, 7, None, None, None, None)
    svc._tab_cache[key] = {"meta": {"care": {"tabs": []}}, "conditions": {"x": 1}}
    assert svc.cached(area="L_6d", severity=0.9, disorder="stroke", start_day=7)
    out = svc.simulate(area="L_6d", severity=0.9, disorder="stroke", start_day=7)
    assert out["conditions"] == {"x": 1}
    assert [t["name"] for t in out["meta"]["care"]["tabs"]] == [t["name"] for t in D.tabs()]


def test_anything_the_user_changed_is_not_cached():
    svc = _bare_service()
    for extra in ({"overrides": {"lesion.cross_spread": 0.5}}, {"preset": "left_6d_stroke"},
                  {"patient": {"damage_map": {"L_4": 0.5}}}, {"judge": {"mode": "bci"}}):
        kw = dict(area="L_6d", severity=0.9, disorder="stroke", preset=None, start_day=7,
                  judge=None, prediction=None, overrides=None, patient=None)
        kw.update(extra)
        assert svc._tab_key(**kw) is None


def test_warm_tabs_simulates_each_tab_once(monkeypatch):
    svc = _bare_service()
    calls = []

    def fake_simulate(**kw):
        calls.append(kw["disorder"])
        key = svc._tab_key(kw["area"], kw["severity"], kw["disorder"], None, kw["start_day"], None, None, None, None)
        svc._tab_cache[key] = {"meta": {"care": {}}}
        return svc._tab_cache[key]

    monkeypatch.setattr(svc, "simulate", fake_simulate)
    names = [t["name"] for t in D.tabs()]
    assert svc.warm_tabs() == names
    assert svc.warm_tabs() == []          # already warm: nothing runs again
    assert calls == names


def test_a_prediction_for_another_area_is_ignored(monkeypatch):
    """The page may send the AI answer for the area clicked before; the server must not plan with it."""
    from brain_model import care_service as cs
    svc = _bare_service()
    svc.base, svc.care = load_config("configs/hcp.yaml"), load_config("configs/care.yaml")
    seen = {}

    def fake_predict(area, severity=0.9, disorder="stroke"):
        seen["area"] = area
        return {"area": area, "best": {"mode": "physical", "chance": 0.5}, "helpers": [], "modes": [], "why": []}

    monkeypatch.setattr(svc, "predict", fake_predict)
    class Planned(Exception):
        pass

    def stop(cfg, pred, judge=None):
        raise Planned(pred)

    monkeypatch.setattr(svc, "plans", stop)   # stop before the slow simulation: we only check the prediction
    stale = {"area": "R_FFC", "best": {"mode": "bci", "chance": 0.9}, "helpers": [{"area": "L_FFC"}]}
    with pytest.raises(Planned) as got:
        svc.simulate(area="L_6d", severity=0.9, disorder="stroke", prediction=stale)
    assert seen["area"] == "L_6d" and got.value.args[0]["area"] == "L_6d"
