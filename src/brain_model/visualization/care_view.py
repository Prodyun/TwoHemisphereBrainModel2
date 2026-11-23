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
    # The help signal on each pathway (helper -> damaged area), frame by frame, so the viewer can show
    # it travelling: gain into the receiver x callosal route weight x the helper's output that moves
    # with the practised pattern. Two-step routes have no direct fibre, so they use the helper's whole
    # route row into the damaged side (its strongest receiver) as the carrier.
    route = b.callosum.into_left_wiring if side == "left" else b.callosum.into_right_wiring
    n_areas = len(names[side])
    R = np.eye(n_areas) if route is None else route
    arcs = [[hp["index"], r, hp["share"]] for hp in help_list[:6] for r in hp["reaches"][:2]]
    raw = {}
    for key, _, _ in plans:
        run = results[key]["run"]
        sent = (1.0 - run[f"{healthy_side}_damage"]) * run[healthy_side]
        gain = run[f"gain_into_{side}"]
        pattern = run[f"{side}_task:{job}"]
        flux = np.zeros((len(sent), len(arcs)))
        for a, (j, i, _) in enumerate(arcs):
            weight = R[i, j] if R[i, j] > 0 else R[:, j].max()
            flux[:, a] = gain[:, i] * weight * np.maximum(sent[:, j] * pattern[:, i], 0.0)
        raw[key] = flux
    top = max([f.max() for f in raw.values()] + [1e-9])
    for key, flux in raw.items():
        frames = [(flux[f * stride:(f + 1) * stride].max(axis=0) / top if len(flux[f * stride:(f + 1) * stride])
                   else np.zeros(len(arcs))) for f in range(n_frames)]
        payload["conditions"][key]["help_flux"] = [[round(float(v), 3) for v in row] for row in frames]
    payload["meta"]["care"] = {
        "arcs": [[healthy_side, int(j), int(i), float(s)] for j, i, s in arcs],
        "penumbra_days": main["lesion"].get("penumbra_steps", 0) / steps_per_day(main),
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
    from .viewer import area_info
    payload["area_info"] = area_info(main_cfg)
    return viewer_html(payload)
