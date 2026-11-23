"""Training examples for the AI Help Predictor: one simulated patient -> one row.

A case = (damaged area, severity, disorder, practice mode, seed). Each case is simulated twice,
with the corpus callosum connected and cut (split brain), using the same noise:
  help     = final function connected - final function split  (what the other side adds)
  success  = the trained job got back to the goal (connected)
plus weeks / physical reps to the goal, final function, and (for the reference mode) the top
helper areas from the pathway tracer. Workers receive only the small case tuple and build the
configs themselves, so nothing large is copied between processes.
"""
from functools import lru_cache

from .help_pathways import trace
from .patient import patient_config, patient_from_area
from .planning import healthy_for, variant
from .simulation.outcome import run_patient
from .utils import load_config

MODES = {"physical": "physical", "imagery": "imagery", "bci": "bci", "bci+vns": ["bci", "vns"],
         "imagery+sleep": ["imagery", "sleep_replay"]}
REFERENCE_MODE = "physical"


@lru_cache(maxsize=2)
def _configs(base_path: str, care_path: str) -> tuple[dict, dict]:
    return load_config(base_path), load_config(care_path)


def case_config(case: tuple, base_path="configs/hcp.yaml", care_path="configs/care.yaml") -> dict:
    area, severity, disorder, mode, seed = case
    base, care = _configs(base_path, care_path)
    cfg = patient_config(base, care, patient_from_area(area, severity, disorder))
    return variant(cfg, seed=seed, therapy={"mode": MODES[mode]})


def case_key(case: tuple) -> str:
    return "|".join(str(c) for c in case)


def run_case(case: tuple) -> dict:
    try:
        return _run_case(case)
    except Exception as e:  # noqa: BLE001 - record and keep going; the trainer skips error rows
        return {"key": case_key(case), "area": case[0], "error": f"{type(e).__name__}: {e}"}


def _run_case(case: tuple) -> dict:
    area, severity, disorder, mode, seed = case
    cfg = case_config(case)
    m, run = run_patient(cfg, healthy_for(cfg))
    split_cfg = variant(cfg, corpus_callosum={"strength": 0.0})
    ms, _ = run_patient(split_cfg, healthy_for(split_cfg))
    row = {"key": case_key(case), "area": area, "severity": severity, "disorder": disorder, "mode": mode,
           "seed": seed, "job": cfg["therapy"]["function"],
           "final_function": m["final_function"], "split_final_function": ms["final_function"],
           "help": m["final_function"] - ms["final_function"],
           "success": m["weeks_to_goal"] is not None, "never_impaired": m["never_impaired"],
           "weeks_to_goal": m["weeks_to_goal"], "physical_reps_to_goal": m["physical_reps_to_goal"],
           "mental_reps_to_goal": m["mental_reps_to_goal"], "lowest_function": m["lowest_function"],
           "weekly_function": m["weekly_function"]}
    if mode == REFERENCE_MODE:
        t = trace(cfg, run)
        row["helpers"] = [{k: h[k] for k in ("area", "share", "route", "via", "reaches")} for h in t["helpers"][:6]]
    return row
