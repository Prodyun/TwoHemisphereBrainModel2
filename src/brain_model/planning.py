"""RECRUIT / DOSE / TIME: building blocks for a patient's personal plan.

  candidate_targets  healthy-hemisphere areas most able to reach the damaged areas: direct callosal
                     fibres into them (plus the matching-area link), and one step further through the
                     healthy side's own wiring. This pre-screen keeps the search fast; every candidate
                     is then really simulated.
  standard_target    today's usual healthy-side target for the trained job (configs/care.yaml).
  variant / evaluate a plan = the patient config + changes; evaluate() runs it and returns the
                     three measures (physical reps to goal, weeks to goal, final function).
  rank               order plans: those that reach the goal first (fewest physical reps, then fewest
                     weeks), then by final function.
"""
import json
from multiprocessing import Pool

import numpy as np

from .brain import Brain
from .patient import deep_merge
from .simulation.outcome import healthy_reference, run_patient


def _sides(cfg: dict):
    brain = Brain(cfg, with_lesion=True)
    brain.lesion._start()
    damaged = brain.left if cfg["lesion"]["hemisphere"] == "left" else brain.right
    healthy = brain.right if damaged is brain.left else brain.left
    return brain, damaged, healthy


def reach_scores(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """How strongly each healthy-side area can send to the damaged areas (weighted by their damage)."""
    brain, damaged, healthy = _sides(cfg)
    n = len(damaged.x)
    cc = brain.callosum
    route = cc.into_left_wiring if damaged is brain.left else cc.into_right_wiring
    C = np.eye(n) if route is None else route.copy()       # receiver (damaged) x sender (healthy)
    C = C + 0.5 * np.eye(n) * (C[C > 0].mean() if (C > 0).any() else 1.0)  # matching-area link (missed by tractography)
    two_step = C @ healthy.W                                # via one healthy-side neighbour
    reach = damaged.damage @ (C + 0.5 * two_step)
    return reach, healthy.region_names


def candidate_targets(cfg: dict, k: int = 20) -> list[str]:
    reach, names = reach_scores(cfg)
    return [names[i] for i in np.argsort(reach)[::-1][:k] if reach[i] > 0]


def standard_target(cfg: dict, care: dict) -> dict:
    job = cfg["therapy"]["function"]
    std = care["standard_targets"].get(job, care["standard_targets"]["default"])
    healthy_prefix = "R_" if cfg["lesion"]["hemisphere"] == "left" else "L_"
    if std["area"] == "mirror":
        _, damaged, _ = _sides(cfg)
        epicentre = damaged.region_names[int(np.argmax(damaged.damage))]
        area = healthy_prefix + epicentre[2:]
    else:
        area = healthy_prefix + str(std["area"])
    return {"area": area, "strength": std["strength"], "why": std["why"]}


def variant(cfg: dict, **changes) -> dict:
    return deep_merge(cfg, changes)


def with_targets(cfg: dict, targets: list[dict]) -> dict:
    return variant(cfg, stimulation={"targets": targets})


_HEALTHY: dict = {}


def healthy_for(plan: dict) -> dict:
    """The healthy reference run for a plan (cached per process: it depends only on the brain,
    the noise seed, the run length and the callosum, not on the lesion or the care)."""
    key = (plan["seed"], plan["steps"], json.dumps(plan["corpus_callosum"], sort_keys=True),
           json.dumps([plan["left_hemisphere"].get("functions"), plan["right_hemisphere"].get("functions")],
                      sort_keys=True, default=str))
    if key not in _HEALTHY:
        if len(_HEALTHY) > 4:
            _HEALTHY.clear()
        _HEALTHY[key] = healthy_reference(plan)
    return _HEALTHY[key]


def mode_label(mode) -> str:
    return mode if isinstance(mode, str) else " + ".join(mode)


def evaluate(job: tuple) -> dict:
    """job = (label, kind, plan config). Safe to use in a worker process."""
    label, kind, plan = job
    m, _ = run_patient(plan, healthy_for(plan))
    stim = plan.get("stimulation", {}).get("targets", [])
    return {"plan": label, "kind": kind, "therapy": mode_label(plan["therapy"]["mode"]),
            "targets": "; ".join(f"{t['area']} {float(t['strength']):+.2f}" for t in stim) or "none",
            "start_day": plan["therapy"].get("start_day"),
            "callosum": plan["corpus_callosum"]["strength"], **m}


def run_plans(jobs: list[tuple], workers: int = 1) -> list[dict]:
    if workers <= 1 or len(jobs) <= 1:
        return [evaluate(j) for j in jobs]
    with Pool(workers) as pool:
        return list(pool.imap(evaluate, jobs, chunksize=1))


# ---------- RECRUIT: where should the healthy side be emphasised? ----------

def search_targets(cfg: dict, care: dict, workers: int = 1, k: int = 20,
                   strengths: tuple = (0.5, -0.3), n_random: int = 3, seed: int = 0) -> list[dict]:
    """Try each pre-screened healthy-side area (excited and calmed) with the patient's practice, and
    compare with no stimulation, today's standard target, random targets and the split brain."""
    jobs = [("practice only", "none", cfg)]
    std = standard_target(cfg, care)
    jobs.append((f"standard: {std['area']} {std['strength']:+}", "standard",
                 with_targets(cfg, [{"area": std["area"], "strength": std["strength"]}])))
    cands = candidate_targets(cfg, k)
    for area in cands:
        for s in strengths:
            jobs.append((f"{area} {s:+}", "candidate", with_targets(cfg, [{"area": area, "strength": s}])))
    _, _, healthy = _sides(cfg)
    others = [a for a in healthy.region_names if a not in cands]
    rng = np.random.default_rng(seed)
    for area in rng.choice(others, size=min(n_random, len(others)), replace=False):
        jobs.append((f"random: {area} {strengths[0]:+}", "random",
                     with_targets(cfg, [{"area": str(area), "strength": strengths[0]}])))
    jobs.append(("split brain (control)", "split", variant(cfg, corpus_callosum={"strength": 0.0})))
    return run_plans(jobs, workers)


def best_targets(rows: list[dict], n: int = 3) -> list[dict]:
    cands = sorted([r for r in rows if r["kind"] == "candidate"], key=rank_key)
    return cands[:n]


def parse_targets(text: str) -> list[dict]:
    if text == "none":
        return []
    out = []
    for part in text.split("; "):
        area, strength = part.rsplit(" ", 1)
        out.append({"area": area, "strength": float(strength)})
    return out


# ---------- DOSE: which practice, how much stimulation ----------

def search_dose(cfg: dict, targets: list[dict], modes: list, strengths: list[float],
                workers: int = 1) -> list[dict]:
    jobs = []
    for mode in modes:
        plan = variant(cfg, therapy={"mode": mode})
        jobs.append((f"{mode_label(mode)}, no stimulation", "dose", plan))
        for s in strengths:
            scaled = [{"area": t["area"], "strength": np.sign(t["strength"]) * s} for t in targets]
            jobs.append((f"{mode_label(mode)}, stimulation {s:.2f}", "dose", with_targets(plan, scaled)))
    return run_plans(jobs, workers)


# ---------- TIME: the golden period ----------

def search_timing(cfg: dict, start_days: list[int], workers: int = 1) -> list[dict]:
    """The same plan started on different days after the injury (stimulation starts with practice)."""
    jobs = []
    for d in start_days:
        jobs.append((f"start day {d}", "timing",
                     variant(cfg, therapy={"start_day": d}, stimulation={"start_day": d})))
    return run_plans(jobs, workers)


def rank_key(row: dict) -> tuple:
    reached = row["weeks_to_goal"] is not None
    return (not reached,
            row["physical_reps_to_goal"] if reached else 0.0,
            row["weeks_to_goal"] if reached else 0.0,
            -row["final_function"])
