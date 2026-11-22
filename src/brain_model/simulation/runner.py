"""Runs the brain over time and runs the full lesion experiment.

The experiment (the science-fair comparison):
  1. Connected brain:  healthy baseline run vs. lesion run, corpus callosum ON.
  2. Split brain:      healthy baseline run vs. lesion run, corpus callosum OFF (the control).
Every run uses the same random noise, so the only differences are the lesion and the callosum.

"Function score" = how strongly a hemisphere's regions still carry their task signal,
compared with the same hemisphere when healthy (1 = 100% of healthy, 0 = task signal gone).
For each region we measure how much its activity rises and falls with its task input
(a regression slope), sum over regions, and divide by the healthy run's value.
"""
import numpy as np

from ..brain import Brain


def run_simulation(config: dict, with_lesion: bool = True) -> dict:
    brain = Brain(config, with_lesion=with_lesion)
    n, steps = config["n_regions"], config["steps"]
    rng = np.random.default_rng(config["seed"] + 1)
    # Same random numbers in every run; each hemisphere scales them by its own noise level.
    noise = rng.normal(0, 1, size=(2, steps, n))
    noise[0] *= brain.left.noise
    noise[1] *= brain.right.noise
    history: dict[str, list] = {}
    for t in range(steps):
        for key, value in brain.step(t, noise[0, t], noise[1, t]).items():
            history.setdefault(key, []).append(value)
    out = {k: np.array(v) for k, v in history.items()}
    out["lesion_center"] = brain.lesion.center_region if brain.lesion else -1
    out["functions"] = {h.name: {f.name: f.regions for f in h.functions} for h in (brain.left, brain.right)}
    return out


def function_scores(lesioned: dict, healthy: dict, side: str, window: int) -> dict[str, np.ndarray]:
    """Function score of each job separately, using only that job's areas and its own input."""
    out = {}
    for name, regions in lesioned["functions"][side].items():
        task = lesioned[f"{side}_task:{name}"][:, regions]
        out[name] = function_score(lesioned[side][:, regions], healthy[side][:, regions], task, window)
    return out


def task_signal(activity: np.ndarray, task: np.ndarray) -> float:
    """Total task signal: sum over regions of slope(activity ~ task). Arrays are (time, regions)."""
    a = activity - activity.mean(axis=0)
    s = task - task.mean(axis=0)
    var = (s ** 2).sum(axis=0)
    ok = var > 0
    return float(((a * s).sum(axis=0)[ok] / var[ok]).sum())


def function_score(activity: np.ndarray, baseline: np.ndarray, task: np.ndarray,
                   window: int) -> np.ndarray:
    """Task signal in `activity` divided by task signal in the healthy `baseline`, per time window."""
    scores = []
    for start in range(0, len(activity) - window + 1, window):
        sl = slice(start, start + window)
        healthy = task_signal(baseline[sl], task[sl])
        scores.append(task_signal(activity[sl], task[sl]) / healthy if healthy > 0 else 0.0)
    return np.array(scores)


def rolling_function_score(activity: np.ndarray, baseline: np.ndarray, task: np.ndarray,
                           window: int, stride: int) -> list:
    """Function score over the trailing `window` steps, at every `stride`-th step (None until full)."""
    out = []
    for t in range(0, len(activity), stride):
        if t + 1 < window:
            out.append(None)
            continue
        sl = slice(t + 1 - window, t + 1)
        healthy = task_signal(baseline[sl], task[sl])
        out.append(task_signal(activity[sl], task[sl]) / healthy if healthy > 0 else 0.0)
    return out


def with_strength(config: dict, strength: float) -> dict:
    return {**config, "corpus_callosum": {**config["corpus_callosum"], "strength": strength}}


def run_experiment(config: dict) -> dict:
    """Run connected vs split-brain conditions and score both hemispheres in each."""
    window = config["analysis"]["window"]
    conditions = {"connected": config, "split": with_strength(config, 0.0)}
    results = {}
    for name, cfg in conditions.items():
        healthy = run_simulation(cfg, with_lesion=False)
        lesioned = run_simulation(cfg, with_lesion=True)
        results[name] = {
            "run": lesioned,
            "healthy": healthy,
            "score": {side: function_score(lesioned[side], healthy[side],
                                           lesioned[f"{side}_task"], window)
                      for side in ("left", "right")},
            "function_scores": {side: function_scores(lesioned, healthy, side, window)
                                for side in ("left", "right")},
        }
    return results


def summarize(results: dict, config: dict) -> dict:
    """Average function score after the lesion has had time to spread (second half of the run)."""
    damaged = config["lesion"]["hemisphere"]
    healthy_side = "right" if damaged == "left" else "left"
    summary = {"damaged_hemisphere": damaged}
    for name, res in results.items():
        late = len(res["score"][damaged]) // 2
        summary[name] = {
            "damaged_side_score": float(res["score"][damaged][late:].mean()),
            "healthy_side_score": float(res["score"][healthy_side][late:].mean()),
            "final_mean_damage": float(res["run"][f"{damaged}_damage"][-1].mean()),
            "functions": {side: {f: float(s[len(s) // 2:].mean()) for f, s in scores.items()}
                          for side, scores in res["function_scores"].items()},
        }
    summary["benefit_of_callosum"] = (summary["connected"]["damaged_side_score"]
                                      - summary["split"]["damaged_side_score"])
    summary["function_benefit"] = {
        f: summary["connected"]["functions"][damaged][f] - summary["split"]["functions"][damaged][f]
        for f in summary["connected"]["functions"][damaged]}
    return summary
