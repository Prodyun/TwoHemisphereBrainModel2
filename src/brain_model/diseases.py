"""The DISEASE REGISTRY: one profile file per disease in configs/diseases/*.yaml (one explorer tab each).

A profile says how the disease behaves in the model and how to explain it:

  name, tab, icon, order       identity; `tab: null` hides it from the tab bar
  engine:                      lesion settings (lesion.py): type stroke | tumor | spreading | epilepsy, + params
  seed:                        where the damage starts
      mode: clicked            the clicked area (+ `neighbors` strongest same-side links at `neighbor_share`)
      mode: areas              fixed areas, e.g. {EC: 0.5, PeEc: 0.4} on `sides` (left/right/both, with scale)
      diffuse: {level, share}  plus small damage on a random share of ALL areas (traumatic brain injury)
  default_area, severity       what the tab starts with
  stages:                      optional named groups of areas (e.g. Braak stages): when each is reached
  extra_jobs:                  jobs added to both hemispheres (e.g. memory); their areas must be free
  job:                         the job to train and score (default: the job of the most damaged area)
  config:                      any config overrides (e.g. care.days, callosal_injury, therapy)
  plans:                       the treatments to compare (always plus the split-brain control)
  ai: true|false               is the AI Help Predictor trained on this disease?
  sliders, plain, clinician, assumptions, sources   for the tab (plain = simple view)

Placeholders in plans: "@epicentre" (the most damaged area), "@mirror" (its healthy-side partner).
`custom: true` marks a profile made with the Disease Builder (shown as "uncited" until it has sources).
"""
import copy
from pathlib import Path

import numpy as np
import yaml

FOLDER = Path("configs/diseases")
ENGINE_TYPES = ("stroke", "tumor", "spreading", "epilepsy")
REQUIRED = ("name", "engine")


def validate(p: dict, where: str = "profile") -> dict:
    for key in REQUIRED:
        if key not in p:
            raise ValueError(f"{where}: missing '{key}'")
    if p["engine"].get("type") not in ENGINE_TYPES:
        raise ValueError(f"{where}: engine.type must be one of {ENGINE_TYPES}")
    seed = p.setdefault("seed", {"mode": "clicked"})
    if seed.get("mode", "clicked") not in ("clicked", "areas"):
        raise ValueError(f"{where}: seed.mode must be 'clicked' or 'areas'")
    if seed.get("mode") == "areas" and not seed.get("areas"):
        raise ValueError(f"{where}: seed.mode 'areas' needs seed.areas")
    sev = float(p.get("severity", 0.9))
    if not 0.05 <= sev <= 0.98:
        raise ValueError(f"{where}: severity must be between 0.05 and 0.98")
    p.setdefault("tab", p["name"])
    p.setdefault("ai", False)
    p.setdefault("plans", [])
    p.setdefault("sources", [])
    p.setdefault("order", 99)
    return p


def load_profiles(folder: Path | str = FOLDER) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(folder).glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            p = yaml.safe_load(fh) or {}
        p = validate(p, str(f))
        out[p["name"]] = p
    return out


_CACHE: dict = {}


def profiles(folder: Path | str = FOLDER, reload: bool = False) -> dict[str, dict]:
    key = str(folder)
    if reload or key not in _CACHE:
        _CACHE[key] = load_profiles(folder) if Path(folder).exists() else {}
    return _CACHE[key]


def get(name: str, folder: Path | str = FOLDER) -> dict | None:
    return profiles(folder).get(name)


def tabs(folder: Path | str = FOLDER) -> list[dict]:
    """The visible tabs in order, with what the explorer needs to show the card."""
    out = []
    for p in sorted(profiles(folder).values(), key=lambda p: (p["order"], p["name"])):
        if p.get("tab") is None:
            continue
        out.append({k: p.get(k) for k in ("name", "tab", "icon", "ai", "custom", "default_area", "severity",
                                          "plain", "clinician", "assumptions", "sources", "sliders", "stages",
                                          "seed", "engine")})
    return out


# ---------- a patient for a disease ----------

def _area_list(names: list[str], side: str, short: str) -> str:
    return ("L_" if side == "left" else "R_") + str(short).removeprefix("L_").removeprefix("R_")


def make_patient(profile: dict, area: str | None = None, severity: float | None = None,
                 anatomy_file: str | None = None, seed: int = 0) -> dict:
    """The damage map for this disease (area names -> 0..1)."""
    from .hcp import DEFAULT_FILE, load_hcp_brain
    from .patient import patient_from_area
    sev = float(severity if severity is not None else profile.get("severity", 0.9))
    s = profile.get("seed", {})
    if s.get("mode", "clicked") == "clicked":
        area = area or profile.get("default_area")
        if not area:
            raise ValueError(f"{profile['name']}: click an area (no default_area)")
        damage = patient_from_area(area, sev, profile["name"], neighbors=int(s.get("neighbors", 4)),
                                   neighbor_share=float(s.get("neighbor_share", 0.6)),
                                   anatomy_file=anatomy_file or DEFAULT_FILE)["damage"]
    else:
        sides = s.get("sides", {"left": 1.0})
        if isinstance(sides, str):
            sides = {"left": 1.0, "right": 1.0} if sides == "both" else {sides: 1.0}
        if area:  # a clicked area moves the seed pattern to that side
            clicked = "left" if area.startswith("L_") else "right"
            other = "right" if clicked == "left" else "left"
            hi, lo = max(sides.values()), min(sides.values()) if len(sides) > 1 else 0.0
            sides = {clicked: hi, **({other: lo} if lo > 0 else {})}
        damage = {}
        for side, scale in sides.items():
            for short, value in s["areas"].items():
                damage[_area_list([], side, short)] = round(min(0.98, sev * float(value) * float(scale)), 3)
    diffuse = s.get("diffuse")
    if diffuse:
        brain = load_hcp_brain(anatomy_file or DEFAULT_FILE)
        rng = np.random.default_rng(seed)
        names = list(brain.names)
        k = int(round(float(diffuse.get("share", 0.3)) * len(names)))
        for nm in rng.choice(names, size=k, replace=False):
            damage[str(nm)] = max(damage.get(str(nm), 0.0), round(float(diffuse.get("level", 0.1)) * sev, 3))
    epic = max(damage, key=damage.get)
    return {"name": f"{profile.get('tab', profile['name'])} ({epic}, severity {sev:.2f})",
            "disorder": profile["name"], "damage": damage}


# ---------- profile -> model config ----------

def engine_settings(profile: dict) -> dict:
    eng = copy.deepcopy(profile["engine"])
    return eng


def add_jobs(cfg: dict, profile: dict) -> dict:
    """Add the profile's extra jobs (e.g. memory) to both hemispheres, skipping areas already in use."""
    for side in ("left", "right"):
        funcs = cfg[f"{side}_hemisphere"]["functions"]
        used = {str(a) for f in funcs.values() for a in f["areas"]}
        for name, job in (profile.get("extra_jobs") or {}).items():
            if name in funcs:
                continue
            areas = [a for a in job["areas"] if str(a) not in used]
            if areas:
                funcs[name] = {**copy.deepcopy(job), "areas": areas}
                used.update(str(a) for a in areas)
    return cfg


def resolve(value, epicentre: str, healthy_prefix: str):
    """Replace @epicentre / @mirror in a plan's settings."""
    if isinstance(value, dict):
        return {k: resolve(v, epicentre, healthy_prefix) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, epicentre, healthy_prefix) for v in value]
    if value == "@epicentre":
        return epicentre
    if value == "@mirror":
        return healthy_prefix + epicentre[2:]
    return value


def plan_list(cfg: dict, profile: dict) -> list[tuple[str, str, dict]]:
    """The profile's treatments for this patient, plus the split-brain control of the first one."""
    from .patient import deep_merge
    side = cfg["lesion"]["hemisphere"]
    on_side = {k: v for k, v in cfg["lesion"]["damage_map"].items() if k.startswith("L_" if side == "left" else "R_")}
    epicentre = max(on_side, key=on_side.get)
    healthy_prefix = "R_" if side == "left" else "L_"
    out = []
    for p in profile.get("plans", []):
        changes = resolve(p.get("changes", {}), epicentre, healthy_prefix)
        out.append((p["key"], p["label"], deep_merge(cfg, changes)))
    if not out:
        out.append(("standard_care", "Standard care: practice", cfg))
    if not any(k == "split" for k, _, _ in out):
        first_key, first_label, first = out[0]
        out.append(("split", f"{first_label.split(':')[0]}, split brain (control)",
                    deep_merge(first, {"corpus_callosum": {"strength": 0.0}})))
    return out


# ---------- disease-specific readouts ----------

def stage_timeline(run: dict, cfg: dict, profile: dict, threshold: float = 0.1) -> list[dict]:
    """For each named stage: the first day its areas (average, both sides) pass `threshold` damage."""
    from .brain import steps_per_day
    from .hcp import load_hcp_brain
    stages = profile.get("stages") or []
    if not stages:
        return []
    brain = load_hcp_brain(cfg["anatomy"]["file"])
    spd, onset = steps_per_day(cfg), cfg["lesion"]["start_step"]
    out = []
    for st in stages:
        cols = {"left": [], "right": []}
        for side in ("left", "right"):
            names = brain.side_names(side)
            pre = "L_" if side == "left" else "R_"
            cols[side] = [names.index(pre + str(a)) for a in st["areas"] if pre + str(a) in names]
        dmg = np.concatenate([run[f"{s}_damage"][:, cols[s]] for s in ("left", "right") if cols[s]], axis=1)
        hit = np.nonzero(dmg.mean(axis=1) >= threshold)[0]
        day = None if not len(hit) else round(float((hit[0] - onset) / spd), 1)
        out.append({"stage": st["name"], "day": day, "areas": [str(a) for a in st["areas"]]})
    return out


def seizure_spread(run: dict, cfg: dict, k: int = 10) -> dict | None:
    """How much of a seizure crosses the bridge.

    A seizure drive set at step t acts on step t+1. For each seizure we take the push it gives:
    activity change from t to t+1, minus the average change on quiet steps. The other side's push is
    averaged over the focus's `k` strongest callosal partners; the own side's over its `k` strongest
    same-side neighbours. `crosses` = other-side push / focus push (0 = stays put)."""
    if "seizure" not in run:
        return None
    from .brain import Brain
    sz = np.asarray(run["seizure"], bool)
    side = cfg["lesion"]["hemisphere"]
    other = "right" if side == "left" else "left"
    starts = int((sz & ~np.concatenate([[False], sz[:-1]])).sum())
    if not sz.any():
        return {"seizures": 0, "crosses": 0.0, "own_spread": 0.0}
    b = Brain(cfg, with_lesion=True)
    focus = b.lesion.center_region
    target = b.left if side == "left" else b.right
    into_other = b.callosum.base_weights(other)          # receiver (other side) x sender (seizing side)
    partners = [i for i in np.argsort(into_other[:, focus])[::-1][:k] if into_other[i, focus] > 0] or [focus]
    neighbours = [i for i in np.argsort(target.W[focus])[::-1][:k] if target.W[focus, i] > 0]
    T = len(sz)
    t_sz = np.nonzero(sz[:-1])[0]
    quiet = np.nonzero(~sz[:-1] & ~np.concatenate([[False], sz[:-2]]) &
                       (np.arange(T - 1) >= cfg["lesion"]["start_step"]))[0]
    def push(hemi: str, cols, lag: int) -> float:
        """Change over `lag` steps after the drive starts acting (signals cross the bridge one step later)."""
        x = run[hemi][:, cols].mean(axis=1)
        ok_sz, ok_q = t_sz[t_sz + lag < T], quiet[quiet + lag < T]
        d_sz = x[ok_sz + lag] - x[ok_sz]
        d_q = x[ok_q + lag] - x[ok_q] if len(ok_q) else np.zeros(1)
        return float(d_sz.mean() - d_q.mean())
    focus_push = push(side, [focus], 1)
    out = {"seizures": starts, "focus_push": round(focus_push, 4),
           "own_spread": round(push(side, neighbours, 2) / focus_push, 4) if focus_push > 0 else 0.0,
           "crosses": round(push(other, partners, 2) / focus_push, 4) if focus_push > 0 else 0.0}
    return out


def integrity_curve(run: dict, cfg: dict) -> list[float] | None:
    if "callosal_integrity" not in run:
        return None
    from .brain import steps_per_day
    spd = steps_per_day(cfg)
    v = np.asarray(run["callosal_integrity"], float)
    week = 7 * spd
    onset = cfg["lesion"]["start_step"]
    return [round(float(v[i:i + week].mean()), 4) for i in range(onset, len(v) - week + 1, week)]


def extra_measures(run: dict, cfg: dict, profile: dict | None) -> dict:
    out = {}
    if profile and profile.get("stages"):
        out["stages"] = stage_timeline(run, cfg, profile)
    sz = seizure_spread(run, cfg)
    if sz is not None:
        out["seizure_spread"] = sz
    ic = integrity_curve(run, cfg)
    if ic is not None:
        out["bridge_integrity_weekly"] = ic
    # damage on both sides at the end (ALS: no healthy side)
    out["final_damage"] = {s: round(float(run[f"{s}_damage"][-1].mean()), 4) for s in ("left", "right")}
    return out


# ---------- the Disease Builder ----------

def save_custom(spec: dict, folder: Path | str = FOLDER) -> Path:
    """Write a user-made profile (validated) as configs/diseases/custom_<name>.yaml."""
    import re
    name = re.sub(r"[^a-z0-9_]+", "_", str(spec.get("name", "")).lower()).strip("_")
    if not name:
        raise ValueError("a custom disease needs a name")
    p = {"name": f"custom_{name}" if not name.startswith("custom_") else name,
         "tab": spec.get("tab") or str(spec.get("name")), "icon": spec.get("icon", "✚"), "order": 90,
         "custom": True, "ai": False, "engine": spec["engine"], "seed": spec.get("seed", {"mode": "clicked"}),
         "default_area": spec.get("default_area"), "severity": float(spec.get("severity", 0.7)),
         "plain": spec.get("plain", {"what": spec.get("what", "A disease made with the Disease Builder.")}),
         "sources": spec.get("sources", []), "assumptions": spec.get("assumptions", ["user-made profile"]),
         "plans": spec.get("plans", [])}
    validate(copy.deepcopy(p), "custom disease")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{p['name']}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(p, fh, sort_keys=False, allow_unicode=True)
    profiles(folder, reload=True)
    return path
