"""A PATIENT: their damage mapped onto the 180 areas of each hemisphere, turned into a model config.

A patient file (YAML) looks like:

    name: "Left 6d stroke"
    disorder: stroke             # any disease profile in configs/diseases/ (stroke, tumor, alzheimers, ...)
    damage: {L_6d: 0.9, L_6a: 0.5, L_4: 0.4}   # area name -> damage (0 healthy .. 1 destroyed)
    therapy: {mode: imagery}     # optional overrides of configs/care.yaml
    biomarkers: {}               # optional (later phases)

`patient_config()` merges configs/hcp.yaml + configs/care.yaml + the patient, and
`problem_card()` is the IDENTIFY step: what is damaged, where, and which jobs are hit.
"""
import copy
from pathlib import Path

import numpy as np
import yaml

from .brain import Brain, steps_per_day
from .hcp import DEFAULT_FILE, load_hcp_brain


def load_patient(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        patient = yaml.safe_load(f)
    if not patient.get("damage"):
        raise ValueError(f"{path}: a patient needs a `damage` map ({{area: 0..1}})")
    patient.setdefault("name", Path(path).stem)
    return patient


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def damaged_side(damage: dict) -> str:
    left = sum(v for k, v in damage.items() if str(k).startswith("L_"))
    right = sum(v for k, v in damage.items() if str(k).startswith("R_"))
    return "left" if left >= right else "right"


def _auto_function(cfg: dict, side: str, damage: dict) -> str:
    """The job of the damaged hemisphere whose areas lost the most (on average)."""
    funcs = cfg[f"{side}_hemisphere"]["functions"]
    prefix = "L_" if side == "left" else "R_"
    best, best_loss = None, -1.0
    for name, f in funcs.items():
        loss = float(np.mean([damage.get(prefix + str(a), 0.0) for a in f["areas"]]))
        if loss > best_loss:
            best, best_loss = name, loss
    return best


def patient_from_area(area: str, severity: float, disorder: str = "stroke", neighbors: int = 4,
                      neighbor_share: float = 0.6, anatomy_file: str = DEFAULT_FILE) -> dict:
    """A patient made from one clicked area: the area at `severity`, plus its `neighbors` most
    strongly wired same-side areas at neighbor_share x severity (a typical lesion is not one atlas area)."""
    brain = load_hcp_brain(anatomy_file)
    side = "left" if area.startswith("L_") else "right"
    names = brain.side_names(side)
    if area not in names:
        raise ValueError(f"unknown area {area}")
    sc = brain.sc_left if side == "left" else brain.sc_right
    row = sc[names.index(area)]
    near = [names[j] for j in np.argsort(row)[::-1][:neighbors] if row[j] > 0]
    damage = {area: float(severity), **{n: round(float(severity) * neighbor_share, 3) for n in near}}
    return {"name": f"{area} {disorder} (severity {severity:.2f})", "disorder": disorder, "damage": damage}


def ensure_job(cfg: dict, side: str, area: str, care: dict) -> str:
    """The job the damaged area belongs to. Many of the 180 areas are not part of any configured job,
    so their recovery could not be scored: they get a LOCAL job (the area plus its strongest same-side
    neighbours not used by other jobs, with a steady rhythm). Returns the job's name."""
    funcs = cfg[f"{side}_hemisphere"]["functions"]
    prefix = "L_" if side == "left" else "R_"
    short = area[2:]
    for name, f in funcs.items():
        if short in [str(a) for a in f["areas"]]:
            return name
    lj = care.get("local_job", {})
    brain = load_hcp_brain(cfg["anatomy"].get("file", DEFAULT_FILE))
    names = brain.side_names(side)
    sc = brain.sc_left if side == "left" else brain.sc_right
    used = {str(a) for f in funcs.values() for a in f["areas"]}
    row = sc[names.index(area)]
    near = [names[j][2:] for j in np.argsort(row)[::-1] if row[j] > 0 and names[j][2:] not in used]
    job = f"local_{short}"
    funcs[job] = {"areas": [short] + near[: lj.get("neighbors", 5)], "pattern": "sine",
                  "params": {"period": lj.get("period", 60), "phase": 0.0}}
    assert all((prefix + a) in names for a in funcs[job]["areas"])
    return job


def patient_config(base: dict, care: dict, patient: dict) -> dict:
    cfg = deep_merge(base, {k: v for k, v in care.items() if k not in ("disorders", "standard_targets")})
    damage = {str(k): float(v) for k, v in patient["damage"].items()}
    side = damaged_side(damage)
    disorder = patient.get("disorder", "stroke")
    from .diseases import add_jobs, engine_settings
    from .diseases import get as get_disease
    from .diseases import profiles as all_profiles
    profile = get_disease(disorder)
    if profile is None and disorder not in care.get("disorders", {}):
        known = sorted(set(all_profiles()) | set(care.get("disorders", {})))
        raise ValueError(f"unknown disorder '{disorder}' (choose from {known})")
    if profile is not None:
        cfg = add_jobs(deep_merge(cfg, profile.get("config", {})), profile)
    spd = steps_per_day(cfg)
    c = cfg["care"]
    onset = int(c["onset_day"] * spd)
    cfg["steps"] = int((c["onset_day"] + c["days"]) * spd)
    cfg["frame_stride"] = max(cfg.get("frame_stride", 4), spd // 2, cfg["steps"] // 300)   # <= ~300 viewer frames
    lesion = {**cfg["lesion"], "hemisphere": side, "region": None, "damage_map": damage,
              "start_step": onset, "cross_spread": 0.0}
    lesion.update(engine_settings(profile) if profile is not None else care["disorders"][disorder])
    if "penumbra_days" in lesion:   # days -> model steps
        lesion["penumbra_steps"] = int(lesion.pop("penumbra_days") * spd)
    cfg["lesion"] = lesion
    cfg = deep_merge(cfg, {k: patient[k] for k in ("therapy", "stimulation", "recovery", "corpus_callosum")
                           if k in patient})
    on_side = {k: v for k, v in damage.items() if k.startswith("L_" if side == "left" else "R_")}
    epicentre = max(on_side, key=on_side.get)
    wanted = (profile or {}).get("job")
    job = wanted if wanted in cfg[f"{side}_hemisphere"]["functions"] else ensure_job(cfg, side, epicentre, care)
    if cfg["therapy"].get("function", "auto") == "auto":
        cfg["therapy"]["function"] = job
    cfg["patient"] = {"name": patient.get("name", "patient"), "disorder": disorder,
                      "biomarkers": patient.get("biomarkers", {})}
    return cfg


def problem_card(cfg: dict) -> dict:
    """IDENTIFY: where the problem is and which jobs it hits, in Glasser area names."""
    brain = Brain(cfg, with_lesion=True)
    brain.lesion._start()
    side = cfg["lesion"]["hemisphere"]
    hemi = brain.left if side == "left" else brain.right
    other = brain.right if side == "left" else brain.left
    jobs = {}
    for h in (hemi, other):
        for f in h.functions:
            loss = float(h.damage[f.regions].mean())
            if loss > 0:
                jobs[f"{h.name}:{f.name}"] = round(loss, 3)
    damaged = {h.region_names[i]: round(float(h.damage[i]), 3)
               for h in (hemi, other) for i in np.argsort(h.damage)[::-1] if h.damage[i] > 0}
    return {
        "name": cfg["patient"]["name"], "disorder": cfg["patient"]["disorder"],
        "damaged_hemisphere": side, "healthy_hemisphere": other.name,
        "epicentre": hemi.region_names[brain.lesion.center_region],
        "damaged_areas": damaged, "jobs_hit": dict(sorted(jobs.items(), key=lambda kv: -kv[1])),
        "trained_job": cfg["therapy"]["function"],
    }
