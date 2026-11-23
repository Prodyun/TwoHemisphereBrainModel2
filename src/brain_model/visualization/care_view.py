"""Viewer data for PATIENT CARE: several plans for one patient side by side (e.g. the AI's
recommended plan, standard care and the split-brain control), plus what the AI predicted and how
the help travels. Used by scripts/explore.py (live) and scripts/patient_viewer.py (a static file).

Each plan is simulated with the same noise. On top of the normal viewer data, every plan gets,
per animation frame: the practice kind (0 none, 1 practice session, 2 night replay), whether
stimulation was on, whether the BCI decoder fired, and the repetition counters; plus its weekly
function curve and its three measures.
"""
import numpy as np

from ..brain import steps_per_day
from ..planning import healthy_for, mode_label
from ..simulation import run_simulation
from ..simulation.outcome import measures
from .viewer import run_payload


def _frames(values, stride: int, n_frames: int, how: str):
    v = np.asarray(values, dtype=float)
    out = []
    for f in range(n_frames):
        chunk = v[f * stride:(f + 1) * stride]
        if not len(chunk):
            chunk = v[-1:]
        out.append(float(chunk.max() if how == "max" else chunk[-1]))
    return out


def _targets(plan: dict, names: dict) -> list:
    out = []
    for t in plan.get("stimulation", {}).get("targets", []):
        side = "left" if t["area"].startswith("L_") else "right"
        out.append([side, names[side].index(t["area"]), float(t["strength"]), t["area"]])
    return out


def care_payload(cfg: dict, plans: list[tuple[str, str, dict]], prediction: dict | None = None,
                 helpers: list | None = None, patient: dict | None = None) -> dict:
    """plans = [(key, label, plan config), ...]; the first plan is the main one."""
    results, plan_meta = {}, {}
    for key, label, plan in plans:
        healthy = healthy_for(plan)
        run = run_simulation(plan, with_lesion=True)
        results[key] = {"run": run, "healthy": healthy}
        plan_meta[key] = {"label": label, "measures": measures(run, healthy, plan),
                          "strength": plan["corpus_callosum"]["strength"],
                          "mode": mode_label(plan["therapy"]["mode"]), "plan": plan}
    main = plans[0][2]
    payload = run_payload(main, results, summary={})
    stride, n_frames = payload["meta"]["stride"], payload["meta"]["n_frames"]
    from ..brain import Brain
    b = Brain(main, with_lesion=True)
    names = {"left": b.left.region_names, "right": b.right.region_names}
    side = main["lesion"]["hemisphere"]
    job = main["therapy"]["function"]
    regions = [int(i) for i in results[plans[0][0]]["run"]["functions"][side][job]]
    for key, _, plan in plans:
        run = results[key]["run"]
        entry = payload["conditions"][key]
        n = len(run["left"])
        entry["practice"] = _frames(run.get("practice", np.zeros(n)), stride, n_frames, "max")
        entry["stimulating"] = _frames(run.get("stimulating", np.zeros(n)), stride, n_frames, "max")
        entry["bci_fired"] = _frames(run.get("bci_fired", np.zeros(n)), stride, n_frames, "max")
        entry["reps_physical"] = _frames(run.get("reps_physical", np.zeros(n)), stride, n_frames, "last")
        entry["reps_mental"] = _frames(run.get("reps_mental", np.zeros(n)), stride, n_frames, "last")
        pm = plan_meta[key]
        entry["weekly"] = pm["measures"]["weekly_function"]
        entry["label"], entry["strength"], entry["mode"] = pm["label"], pm["strength"], pm["mode"]
        entry["measures"] = {k: v for k, v in pm["measures"].items() if k != "weekly_function"}
        entry["targets"] = _targets(plan, names)
    healthy_side = "right" if side == "left" else "left"
    help_list = []
    for h in helpers or []:
        if h["area"] in names[healthy_side]:
            help_list.append({"side": healthy_side, "index": names[healthy_side].index(h["area"]), "area": h["area"],
                              "share": h["share"], "route": h["route"],
                              "reaches": [names[side].index(r) for r in h.get("reaches", []) if r in names[side]]})
    payload["meta"]["care"] = {
        "goal": main["care"]["goal"], "steps_per_day": steps_per_day(main), "onset": main["lesion"]["start_step"],
        "week_steps": int(main["care"]["week_days"] * steps_per_day(main)), "job": job, "job_regions": regions,
        "order": [k for k, _, _ in plans], "patient": patient or {}, "prediction": prediction, "helpers": help_list,
        "damage_map": main["lesion"].get("damage_map", {}),
    }
    return payload


def care_page(main_cfg: dict, payload: dict, interactive: bool = False) -> str:
    """A complete viewer page (meshes included) for a care payload."""
    from .viewer import _mesh_payload, viewer_html
    payload = dict(payload)
    payload["meta"] = {**payload["meta"], "interactive": interactive}
    payload["specialization"] = {s: main_cfg[f"{s}_hemisphere"].get("specialization", s) for s in ("left", "right")}
    payload["mesh"] = {s: _mesh_payload(main_cfg, s) for s in ("left", "right")}
    return viewer_html(payload)
