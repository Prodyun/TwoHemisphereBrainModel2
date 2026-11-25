"""HEALTHY SIDE HELPS: plans in which the healthy hemisphere actively treats the damaged one.

  bridge_builder(cfg)   RECRUIT a pathway: try healthy -> damaged area pairs for paired stimulation
                        (ccPAS) and rank them; controls = the best pair in reverse order, a random
                        sender, and the split brain.
  mirror_drive(cfg)     closed loop: healthy-side activity triggers stimulation of the damaged side;
                        controls = the SAME share of stimulated steps at fixed times and at random times
                        (the share is measured from the closed-loop run), and the split brain.
  help_or_hinder(cfg)   EXCITE, CALM or LEAVE the healthy side (at today's standard target), and why.
  hand_off(cfg, ...)    how to wean the healthy-side help: keep it to the end, stop at week 6, or taper
                        it; scored by INDEPENDENCE = function in the last two weeks, when no help is
                        given any more.

The mechanisms live in connectivity/bridge_therapy.py (Bridge Builder, Mirror Drive) and
therapy.taper_factor (Hand-off). Every result is a model prediction, compared with a control.
"""
import numpy as np

from .help_pathways import trace
from .planning import _sides, rank_key, run_plans, standard_target, variant, with_targets
from .simulation import run_simulation


def _bridge(cfg: dict, care: dict, **changes) -> dict:
    """The patient config with a bridge therapy (defaults from care.yaml `bridge:`)."""
    base = dict(care.get("bridge", {}))
    base.update(start_day=cfg["therapy"].get("start_day", 7))
    return variant(cfg, bridge={**base, **changes})


def _job_damage(cfg: dict) -> dict[str, float]:
    """Damage of each area of the trained job on the damaged side (at the injury)."""
    _, damaged, _ = _sides(cfg)
    job = {f.name: f for f in damaged.functions}[cfg["therapy"]["function"]]
    return {damaged.region_names[i]: float(damaged.damage[i]) for i in job.regions}


def _alive_but_hurt(cfg: dict, k: int = 4, low: float = 0.15, high: float = 0.85) -> list[str]:
    """Damaged job areas that still have living tissue to work with (stimulating dead tissue does nothing)."""
    dmg = _job_damage(cfg)
    hurt = [a for a, d in sorted(dmg.items(), key=lambda kv: -kv[1]) if low <= d <= high]
    return hurt[:k] or sorted(dmg, key=dmg.get, reverse=True)[:k]


def meaningful(cfg: dict) -> float:
    """Smallest change in final function that counts as a real difference (care.yaml care.meaningful_change)."""
    return float(cfg["care"].get("meaningful_change", 0.01))


def better(a: dict, b: dict, tol: float) -> bool:
    """Is plan a really better than plan b? Reaches the goal sooner, or ends >= tol higher."""
    wa, wb = a["weeks_to_goal"], b["weeks_to_goal"]
    if wa is not None and (wb is None or wa < wb):
        return True
    if wa == wb and a["final_function"] >= b["final_function"] + tol:
        return True
    return False


def mirror_of(area: str) -> str:
    return ("R_" if area.startswith("L_") else "L_") + area[2:]


# ---------- Bridge Builder ----------

def bridge_pairs(cfg: dict, helpers: list[dict] | None = None, k: int = 6) -> list[dict]:
    """Candidate pairs: each top helper with the damaged job areas it reaches, plus mirror pairs."""
    if helpers is None:
        helpers = trace(cfg, run_simulation(cfg, with_lesion=True))["helpers"]
    job = _job_damage(cfg)
    pairs, seen = [], set()
    for h in helpers[:4]:
        for r in h.get("reaches", []):
            if r in job and (h["area"], r) not in seen:
                seen.add((h["area"], r))
                pairs.append({"from": h["area"], "to": r})
    for b in _alive_but_hurt(cfg, 3):
        if (mirror_of(b), b) not in seen:
            seen.add((mirror_of(b), b))
            pairs.append({"from": mirror_of(b), "to": b})
    return pairs[:k]


def bridge_builder(cfg: dict, care: dict, helpers: list[dict] | None = None, workers: int = 1,
                   seed: int = 0) -> dict:
    pairs = bridge_pairs(cfg, helpers)
    jobs = [("practice only", "none", cfg)]
    for p in pairs:
        jobs.append((f"{p['from']} -> {p['to']}", "pair", _bridge(cfg, care, mode="pair", pairs=[p])))
    rows = run_plans(jobs, workers)
    cands = sorted([r for r in rows if r["kind"] == "pair"], key=rank_key)
    helps = bool(cands) and better(cands[0], rows[0], meaningful(cfg))
    best = cands[0]["plan"].split(" -> ") if cands else None
    controls = []
    if best:
        bp = {"from": best[0], "to": best[1]}
        _, _, healthy = _sides(cfg)
        rng = np.random.default_rng(seed)
        others = [a for a in healthy.region_names if a not in {p["from"] for p in pairs}]
        rnd = {"from": str(rng.choice(others)), "to": bp["to"]}
        bc = care.get("bridge", {}).get("pair", {})
        controls = run_plans([
            ("best pair, reverse order", "reverse",
             _bridge(cfg, care, mode="pair", pairs=[bp], pair={**bc, "order": "reverse"})),
            (f"random sender {rnd['from']} -> {rnd['to']}", "random", _bridge(cfg, care, mode="pair", pairs=[rnd])),
            ("best pair, split brain", "split",
             variant(_bridge(cfg, care, mode="pair", pairs=[bp]), corpus_callosum={"strength": 0.0})),
        ], workers)
    return {"pairs": pairs, "rows": rows + controls, "best": cands[:3], "none": rows[0], "helps": helps}


# ---------- Mirror Drive ----------

def mirror_pairs(cfg: dict) -> list[dict]:
    return [{"from": mirror_of(b), "to": b} for b in _alive_but_hurt(cfg, 4)]


def measured_duty(cfg: dict) -> float:
    """Share of the practice steps on which the closed loop fired (to give the controls the same dose)."""
    run = run_simulation(cfg, with_lesion=True)
    practice = np.asarray(run["practice"]) == 1
    start = cfg["lesion"]["start_step"] + int(cfg["bridge"].get("start_day", 7) * cfg["time"]["steps_per_day"])
    active = practice & (np.arange(len(practice)) >= start)
    return float(np.asarray(run["bridge_fired"])[active].mean()) if active.any() else 0.0


def mirror_drive(cfg: dict, care: dict, workers: int = 1) -> dict:
    pairs = mirror_pairs(cfg)
    md = dict(care.get("bridge", {}).get("mirror_drive", {}))
    closed = _bridge(cfg, care, mode="mirror_drive", pairs=pairs, mirror_drive={**md, "timing": "closed"})
    duty = measured_duty(closed)
    jobs = [("practice only", "none", cfg),
            ("Mirror Drive (closed loop)", "closed", closed),
            ("same dose, fixed times", "open",
             _bridge(cfg, care, mode="mirror_drive", pairs=pairs, mirror_drive={**md, "timing": "open", "duty": duty})),
            ("same dose, random times", "random",
             _bridge(cfg, care, mode="mirror_drive", pairs=pairs, mirror_drive={**md, "timing": "random", "duty": duty})),
            ("Mirror Drive, split brain", "split", variant(closed, corpus_callosum={"strength": 0.0}))]
    rows = run_plans(jobs, workers)
    by = {r["kind"]: r for r in rows}
    tol = meaningful(cfg)
    return {"pairs": pairs, "duty": duty, "rows": rows,
            "helps": better(by["closed"], by["none"], tol),
            "beats_timing": better(by["closed"], by["open"], tol) and better(by["closed"], by["random"], tol)}


# ---------- Help-or-Hinder ----------

def reserve(cfg: dict) -> dict:
    """The patient's 'reserve' (bimodal balance-recovery model, Di Pino 2014): how much of the trained job
    survives on the damaged side, and how strong the bridge into it is."""
    brain, damaged, _ = _sides(cfg)
    job = {f.name: f for f in damaged.functions}[cfg["therapy"]["function"]]
    spared = float(1.0 - damaged.damage[job.regions].mean())
    w = brain.callosum.base_weights(damaged.name)[job.regions]
    bridge = float(cfg["corpus_callosum"]["strength"] * w.sum(axis=1).mean())
    return {"spared": round(spared, 3), "bridge": round(bridge, 4)}


def help_or_hinder(cfg: dict, care: dict, workers: int = 1, excite: float = 0.5, calm: float = -0.3) -> dict:
    std = standard_target(cfg, care)
    area = std["area"]
    rows = run_plans([
        ("LEAVE the healthy side", "leave", cfg),
        (f"EXCITE {area}", "excite", with_targets(cfg, [{"area": area, "strength": excite}])),
        (f"CALM {area} (today's standard)", "calm", with_targets(cfg, [{"area": area, "strength": calm}])),
    ], workers)
    by = {r["kind"]: r for r in rows}
    tol = meaningful(cfg)
    # Only call it a win if it really changes the outcome (else LEAVE: no stimulation needed)
    best = by["leave"]
    for kind in ("excite", "calm"):
        if better(by[kind], best, tol):
            best = by[kind]
    word = {"leave": "LEAVE", "excite": "EXCITE", "calm": "CALM"}[best["kind"]]
    res = reserve(cfg)
    if word == "EXCITE":
        why = f"Only {res['spared']:.0%} of the job survives on the damaged side, so the healthy side must carry it."
    elif word == "CALM":
        why = f"{res['spared']:.0%} of the job survives: the healthy side's brake is in the way."
    else:
        why = "Changing the healthy side does not change this patient's recovery."
    if not cfg["corpus_callosum"].get("inhibition"):
        why += " (The model's brake is off, so calming cannot help here.)"
    return {"word": word, "area": area, "why": why, "reserve": res, "rows": rows}


# ---------- Hand-off ----------

def hand_off(cfg: dict, targets: list[dict], workers: int = 1) -> dict:
    """Ways to wean the healthy-side stimulation. All schedules stop >= 2 weeks before the end, so the last
    two weeks measure what the damaged side can do ON ITS OWN (the independence score)."""
    days = cfg["care"]["days"]
    stop = days - 14
    start = cfg["therapy"].get("start_day", 7)
    plan = with_targets(cfg, targets)
    schedules = [
        ("help to the end, then stop", {"kind": "step", "to_day": stop}),
        ("stop at week 6", {"kind": "step", "to_day": min(stop, 42)}),
        ("taper weeks 4-10 (straight line)", {"kind": "linear", "from_day": max(start, 28), "to_day": min(stop, 70)}),
        ("taper weeks 4-10 (fast then slow)", {"kind": "exp", "from_day": max(start, 28), "to_day": min(stop, 70)}),
    ]
    jobs = [("no healthy-side help", "none", cfg)]
    jobs += [(name, "schedule", variant(plan, stimulation={"taper": taper})) for name, taper in schedules]
    rows = run_plans(jobs, workers)
    for r in rows:
        r["independence"] = r["final_function"]
    help_days = {name: taper["to_day"] - start for name, taper in schedules}
    for r in rows:
        r["help_days"] = help_days.get(r["plan"], 0)
    sched = [r for r in rows if r["kind"] == "schedule"]
    top = max(sched, key=rank_key_independent) if sched else None
    tol = meaningful(cfg)
    # among the schedules as good as the best (same weeks, independence within tol), the shortest help wins
    good = [r for r in sched if not better(top, r, tol)]
    best = min(good, key=lambda r: r["help_days"]) if good else None
    return {"targets": targets, "rows": rows, "best": best, "helps": bool(best) and better(best, rows[0], tol)}


def rank_key_independent(r: dict) -> tuple:
    return (r["independence"] if r["weeks_to_goal"] is not None else r["independence"] - 1, -(r["weeks_to_goal"] or 99))
