"""PREDICT: how a patient recovers, measured three ways.

  1. physical repetitions needed to get the trained job back to the goal (default 80% of healthy)
  2. weeks needed to reach the goal
  3. final function (average of the last two weeks)

The job's strength is scored week by week with the same "function score" as the rest of the model
(how strongly its areas still carry the job's own signal, compared with the same brain when healthy,
using the same random noise), only on steps when no practice is going on, so practice input itself
is never counted as recovery.
"""
import copy

import numpy as np

from ..brain import steps_per_day
from .runner import run_simulation, task_signal


def without_care(cfg: dict) -> dict:
    """The same patient with no therapy and no stimulation (healing itself stays on)."""
    out = copy.deepcopy(cfg)
    out["therapy"] = {**out.get("therapy", {}), "mode": "none"}
    out["stimulation"] = {**out.get("stimulation", {}), "targets": []}
    return out


def healthy_reference(cfg: dict) -> dict:
    return run_simulation(cfg, with_lesion=False)


def scored_steps(run: dict, washout: int) -> np.ndarray:
    """Steps with no practice and no recent practice."""
    busy = np.asarray(run.get("practice", np.zeros(len(run["left"]))), dtype=int) != 0
    mask = ~busy
    for k in range(1, washout + 1):
        mask &= ~np.concatenate([np.zeros(k, bool), busy[:-k]])
    return mask


def weekly_function(run: dict, healthy: dict, cfg: dict) -> np.ndarray:
    side = cfg["lesion"]["hemisphere"]
    job = cfg["therapy"]["function"]
    regions = run["functions"][side][job]
    c = cfg["care"]
    spd = steps_per_day(cfg)
    onset = cfg["lesion"]["start_step"]
    week = int(c["week_days"] * spd)
    mask = scored_steps(run, c["washout_steps"])
    act, ref = run[side][:, regions], healthy[side][:, regions]
    task = run[f"{side}_task:{job}"][:, regions]
    scores = []
    for start in range(onset, len(act) - week + 1, week):
        sl = np.arange(start, start + week)
        sl = sl[mask[sl]]
        h = task_signal(ref[sl], task[sl])
        scores.append(task_signal(act[sl], task[sl]) / h if h > 0 else 0.0)
    return np.array(scores)


def measures(run: dict, healthy: dict, cfg: dict) -> dict:
    """Recovered = the job's score averages >= goal over two weeks in a row (sustained, and less
    sensitive to how the job's rhythm happens to fall into one week)."""
    scores = weekly_function(run, healthy, cfg)
    goal = cfg["care"]["goal"]
    spd = steps_per_day(cfg)
    onset = cfg["lesion"]["start_step"]
    week = int(cfg["care"]["week_days"] * spd)
    two_week = np.convolve(scores, [0.5, 0.5], mode="full")[1:len(scores)] if len(scores) > 1 else scores
    # Recovery only counts after the job has first fallen below the goal (a slow tumour or a
    # spreading disease can start above it and decline later). Never below = never impaired.
    below = np.nonzero(scores < goal)[0]
    first_dip = int(below[0]) if len(below) else len(scores)
    reached = np.nonzero(two_week >= goal)[0] + 1  # index of the second week of the pair
    reached = reached[reached > first_dip]
    reps_p = np.asarray(run.get("reps_physical", np.zeros(len(run["left"]))), float)
    reps_m = np.asarray(run.get("reps_mental", np.zeros(len(run["left"]))), float)
    out = {"weekly_function": scores.round(4).tolist(), "final_function": float(scores[-2:].mean()),
           "weeks_to_goal": None, "physical_reps_to_goal": None, "mental_reps_to_goal": None,
           "physical_reps_total": float(reps_p[-1]), "mental_reps_total": float(reps_m[-1])}
    if len(reached):
        w = int(reached[0])
        end = min(onset + (w + 1) * week, len(reps_p)) - 1
        out.update(weeks_to_goal=w + 1, physical_reps_to_goal=float(reps_p[end]),
                   mental_reps_to_goal=float(reps_m[end]))
    side = cfg["lesion"]["hemisphere"]
    regions = run["functions"][side][cfg["therapy"]["function"]]
    out["final_damage_trained_areas"] = float(run[f"{side}_damage"][-1][regions].mean())
    out["never_impaired"] = bool(first_dip == len(scores))
    out["lowest_function"] = float(scores.min())
    out["weeks_at_or_above_goal"] = int((scores >= goal).sum())
    return out


def run_patient(cfg: dict, healthy: dict | None = None) -> tuple[dict, dict]:
    """(measures, run). Pass `healthy` (from healthy_reference) to reuse it across many plans."""
    healthy = healthy if healthy is not None else healthy_reference(cfg)
    run = run_simulation(cfg, with_lesion=True)
    return measures(run, healthy, cfg), run
