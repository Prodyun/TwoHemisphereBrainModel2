"""SPREAD RISK: the chance that each of the 360 areas is reached by day 30 / 90 / 180, without treatment.

The disease's spread speed is uncertain, so the spread is run many times (an ensemble), each time with
speeds drawn from a range (profile `uncertainty:`, default 0.5x-1.5x of the profile's values:
ASSUMPTION). For each area: the share of runs in which its damage passed `threshold` by that day.

  spreading (Alzheimer's, Parkinson's, ALS):  the damage front along the wiring. Only the lesion's own
        rule is run (no brain activity, no healing), so it is fast and shows the UNTREATED course. It
        takes one step per DAY with the hourly rates x 24 (an approximation; the rates are small). The
        "next areas" list ranks the areas not yet reached by their average arrival day. A second list
        shows the same if the bridge also carries the disease (cross_spread 0.5).
  stroke / tumour:  the at-risk ring: the areas that are hit but not yet dead (penumbra / pressure);
        their risk = their final damage share (lost unless blood flow returns / the mass is removed).
  callosal injury (MS, TBI):  the chance that each area's incoming bridge fibres fall below 70%.
  epilepsy:  the share of seizures in which each area is pushed by more than 0.05 (recruited).
"""
import copy

import numpy as np

from .brain import Brain, steps_per_day
from .lesion import Lesion

DAYS = (30, 90, 180)


def _names(brain: Brain) -> dict:
    return {"left": brain.left.region_names, "right": brain.right.region_names}


def _pack(prob: dict, names: dict, days, top: int = 10) -> dict:
    """{day: {side: [p per area]}} plus the top at-risk areas at the last day."""
    out = {"days": list(days), "p": {}, "top": []}
    for d in days:
        out["p"][str(d)] = {s: [round(float(v), 3) for v in prob[d][s]] for s in ("left", "right")}
    last = days[-1]
    flat = [(prob[last][s][i], names[s][i]) for s in ("left", "right") for i in range(len(names[s]))]
    out["top"] = [{"area": a, "p": round(float(p), 3)} for p, a in sorted(flat, reverse=True)[:top] if p > 0]
    return out


def spreading_risk(cfg: dict, profile: dict | None = None, n_runs: int = 30, days=DAYS,
                   threshold: float = 0.1, seed: int = 0, cross_spread: float | None = None) -> dict:
    spd = steps_per_day(cfg)
    lcfg = copy.deepcopy(cfg["lesion"])
    if cross_spread is not None:
        lcfg["cross_spread"] = cross_spread
    unc = (profile or {}).get("uncertainty", {})
    lo, hi = unc.get("speed", [0.5, 1.5])
    rng = np.random.default_rng(seed)
    brain = Brain(cfg, with_lesion=True)
    target = brain.lesion.target
    other = brain.left if target is brain.right else brain.right
    routes = ((brain.callosum.route_into_left, brain.callosum.route_into_right) if target is brain.left
              else (brain.callosum.route_into_right, brain.callosum.route_into_left))
    start = lcfg["start_step"]
    n_days = int(max(days))
    reached = {d: {"left": np.zeros(len(brain.left.x)), "right": np.zeros(len(brain.right.x))} for d in days}
    arrival = {"left": np.zeros(len(brain.left.x)), "right": np.zeros(len(brain.right.x))}
    for _ in range(n_runs):
        k = rng.uniform(lo, hi) * spd      # one update per day
        run_cfg = {**lcfg, "start_step": 0, "transport": lcfg.get("transport", 0.0) * k,
                   "growth": lcfg.get("growth", 0.0) * k, "spread_rate": lcfg.get("spread_rate", 0.0) * k}
        target.damage[:] = 0.0
        other.damage[:] = 0.0
        les = Lesion(run_cfg, target, other, *routes)
        first = {"left": np.full(len(target.x), np.inf), "right": np.full(len(target.x), np.inf)}
        for day in range(0, n_days + 1):
            les.update(day)
            for h in (target, other):
                new = (h.damage >= threshold) & np.isinf(first[h.name])
                first[h.name][new] = day
        for d in days:
            for s in ("left", "right"):
                reached[d][s] += first[s] <= d
        for s in ("left", "right"):
            arrival[s] += np.where(np.isinf(first[s]), n_days * 2, first[s])
    prob = {d: {s: reached[d][s] / n_runs for s in ("left", "right")} for d in days}
    names = _names(brain)
    out = _pack(prob, names, days)
    mean_day = {s: arrival[s] / n_runs for s in ("left", "right")}
    nxt = [(mean_day[s][i], names[s][i], prob[days[-1]][s][i]) for s in ("left", "right")
           for i in range(len(names[s])) if mean_day[s][i] > 0.5]   # day 0 = a starting area
    out["next"] = [{"area": a, "day": round(float(d), 1), "p": round(float(p), 3)}
                   for d, a, p in sorted(nxt)[:10]]
    out["top"] = out["next"]
    return out


def ring_risk(cfg: dict) -> dict:
    """Stroke / tumour: the hit-but-alive areas around the centre."""
    brain = Brain(cfg, with_lesion=True)
    les = brain.lesion
    les._start()
    names = _names(brain)
    prob = {"left": np.zeros(len(brain.left.x)), "right": np.zeros(len(brain.right.x))}
    dm = cfg["lesion"].get("damage_map") or {}
    for area, value in dm.items():
        side = "left" if area.startswith("L_") else "right"
        i = names[side].index(area)
        prob[side][i] = float(value)
    centre = les.target.region_names[les.center_region]
    prob[les.target.name][les.center_region] = 0.0     # the core is already lost
    out = _pack({DAYS[-1]: prob}, names, (DAYS[-1],))
    out["core"] = centre
    out["kind"] = "ring"
    return out


def bridge_risk(cfg: dict, n_runs: int = 30, days=DAYS, below: float = 0.7, seed: int = 0) -> dict:
    """MS / TBI: chance that each area's incoming bridge fibres drop below `below` (70%) by each day."""
    from .connectivity.callosal_injury import CallosalInjury
    spd = steps_per_day(cfg)
    brain = Brain(cfg, with_lesion=False)
    names = _names(brain)
    days = [d for d in days if d <= cfg["care"]["days"]] or [cfg["care"]["days"]]
    hits = {d: {"left": np.zeros(len(names["left"])), "right": np.zeros(len(names["right"]))} for d in days}
    for r in range(n_runs):
        cc = Brain(cfg, with_lesion=False).callosum
        inj = CallosalInjury(cfg["callosal_injury"], cc, spd, 0, np.random.default_rng(seed + r))
        low = {"left": np.zeros(len(names["left"]), bool), "right": np.zeros(len(names["right"]), bool)}
        marks = {int(d * spd): d for d in days}
        for t in range(0, int(max(days) * spd) + 1):
            inj.update(t)
            low["left"] |= cc.integrity_into_left < below
            low["right"] |= cc.integrity_into_right < below
            if t in marks:
                for s in ("left", "right"):
                    hits[marks[t]][s] += low[s]
    prob = {d: {s: hits[d][s] / n_runs for s in ("left", "right")} for d in days}
    out = _pack(prob, names, days)
    out["kind"] = "bridge"
    return out


def seizure_risk(cfg: dict, push: float = 0.05) -> dict:
    """Epilepsy: share of seizures in which each area is pushed by more than `push` (recruited)."""
    from .simulation import run_simulation
    plan = copy.deepcopy(cfg)
    plan["therapy"] = {**plan["therapy"], "mode": "none"}
    run = run_simulation(plan, with_lesion=True)
    brain = Brain(plan, with_lesion=True)
    names = _names(brain)
    sz = np.asarray(run["seizure"], bool)
    T = len(sz)
    t_sz = np.nonzero(sz[:-2])[0]
    prob = {}
    for s in ("left", "right"):
        x = run[s]
        if not len(t_sz):
            prob[s] = np.zeros(x.shape[1])
            continue
        d = x[t_sz + 2] - x[t_sz]
        quiet = x[2:] - x[:-2]
        prob[s] = ((d - quiet.mean(axis=0)) > push).mean(axis=0)
    out = _pack({DAYS[0]: prob}, names, (DAYS[0],))
    out["kind"] = "seizure"
    out["seizures"] = int(len(t_sz))
    return out


def risk(cfg: dict, profile: dict | None = None, n_runs: int = 30) -> dict:
    """The right risk map for this disease."""
    kind = cfg["lesion"]["type"]
    if kind == "spreading":
        days = tuple(d for d in DAYS if d <= cfg["care"]["days"]) or (cfg["care"]["days"],)
        out = spreading_risk(cfg, profile, n_runs, days)
        out["kind"] = "spread"
        if not cfg["lesion"].get("cross_spread"):
            out["if_bridge_carries"] = spreading_risk(cfg, profile, n_runs, days, cross_spread=0.5)["next"]
        return out
    if kind == "epilepsy":
        return seizure_risk(cfg)
    if cfg.get("callosal_injury"):
        out = bridge_risk(cfg, n_runs)
        out["ring"] = ring_risk(cfg)["top"]
        return out
    return ring_risk(cfg)
