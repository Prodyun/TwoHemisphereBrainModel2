"""A PERSONAL PLAN for one patient: identify -> recruit -> dose -> time -> predict.

    python scripts/therapy_plan.py --patient configs/patients/left_6d_stroke.yaml
    python scripts/therapy_plan.py --patient <file> --quick        # fewer options, ~1 minute

  IDENTIFY  what is damaged and which jobs are hit (Glasser area names)
  RECRUIT   which healthy-side areas to stimulate (scripts/find_targets.py does this part alone)
  DOSE      which practice (physical, imagery, BCI, sleep replay, VNS, mixes) and how strong
  TIME      the golden period: the same plan started on different days
  PREDICT   the recommended plan vs standard care, today's standard target, no therapy and the
            split brain (control): physical reps to goal, weeks to goal, final function

Output: results/plans/<patient>/personal_plan.md, plan.json, 1_where_to_help.png, 2_recovery.png
Every number is a model prediction under the assumptions listed in docs/factors.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from brain_model.patient import load_patient, patient_config, problem_card  # noqa: E402
from brain_model.planning import (best_targets, mode_label, parse_targets, rank_key, run_plans,  # noqa: E402
                                  search_targets, standard_target, variant, with_targets)
from brain_model.simulation.outcome import without_care  # noqa: E402
from brain_model.utils import load_config  # noqa: E402
from brain_model.visualization.plan_figures import recovery_curves, target_map  # noqa: E402

MODES = ["physical", ["physical", "imagery"], ["physical", "vns"], "imagery", ["imagery", "sleep_replay"],
         "bci", ["bci", "vns"], ["bci", "sleep_replay", "vns"]]
QUICK_MODES = ["physical", ["physical", "vns"], "bci", ["bci", "vns"]]


def target_text(targets):
    return ", ".join(f"{t['area']} ({t['strength']:+.2f})" for t in targets)


def pct(v):
    return "not reached" if v is None else f"{v * 100:.1f}%"


def num(v):
    return "not reached" if v is None else f"{v:,.0f}"


def weeks(v):
    return "not reached" if v is None else str(v)


def rows_table(rows, top=None):
    out = ["| Plan | Final function | Weeks to goal | Physical reps to goal | Mental reps to goal |",
           "|---|---|---|---|---|"]
    for r in rows[:top] if top else rows:
        out.append(f"| {r['plan']} | {pct(r['final_function'])} | {weeks(r['weeks_to_goal'])} | "
                   f"{num(r['physical_reps_to_goal'])} | {num(r['mental_reps_to_goal'])} |")
    return out


def savings(best, ref):
    """Plain-language comparison of the recommended plan with a reference plan."""
    parts = []
    if best["weeks_to_goal"] is not None and ref["weeks_to_goal"] is not None:
        dw = ref["weeks_to_goal"] - best["weeks_to_goal"]
        dr = ref["physical_reps_to_goal"] - best["physical_reps_to_goal"]
        share = dr / ref["physical_reps_to_goal"] * 100 if ref["physical_reps_to_goal"] else 0
        parts.append(f"{dr:,.0f} fewer physical repetitions ({share:.0f}% fewer)")
        parts.append(f"{dw} week(s) sooner" if dw >= 0 else f"{-dw} week(s) later")
    elif best["weeks_to_goal"] is not None:
        parts.append(f"reaches the goal in week {best['weeks_to_goal']}, which {ref['plan']} never does")
    parts.append(f"{(best['final_function'] - ref['final_function']) * 100:+.1f} points of final function")
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--config", default="configs/hcp.yaml")
    ap.add_argument("--care", default="configs/care.yaml")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    care = load_config(args.care)
    patient = load_patient(args.patient)
    cfg = patient_config(load_config(args.config), care, patient)
    out_dir = Path(args.out or f"results/plans/{Path(args.patient).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    k = 10 if args.quick else args.k

    # IDENTIFY
    card = problem_card(cfg)
    print(f"IDENTIFY  {card['name']}: {card['disorder']} in the {card['damaged_hemisphere']} hemisphere, "
          f"epicentre {card['epicentre']}, trained job '{card['trained_job']}'", flush=True)

    # RECRUIT
    print(f"RECRUIT   trying {k} healthy-side areas ...", flush=True)
    target_rows = search_targets(cfg, care, args.workers, k)
    top = best_targets(target_rows, 2)
    single = parse_targets(top[0]["targets"])
    pair = single + parse_targets(top[1]["targets"]) if len(top) > 1 else single
    pair_row = run_plans([("two sites: " + ", ".join(t["area"] for t in pair), "pair", with_targets(cfg, pair))])[0]
    target_rows.append(pair_row)
    targets = pair if rank_key(pair_row) < rank_key(top[0]) else single
    print(f"          best: {target_text(targets)}", flush=True)

    # DOSE
    modes = QUICK_MODES if args.quick else MODES
    strengths = [0.5, 1.0] if args.quick else [0.25, 0.5, 1.0]
    jobs, plans = [], []
    for mode in modes:
        base = variant(cfg, therapy={"mode": mode})
        jobs.append((f"{mode_label(mode)}, no stimulation", "dose", base))
        for s in strengths:
            # excite up to +s; calming is capped at 60% (-0.6) so an area is never switched off
            scaled = [{"area": t["area"], "strength": float(s if t["strength"] > 0 else -0.6 * s)} for t in targets]
            jobs.append((f"{mode_label(mode)} + stimulation {s:.2f}", "dose", with_targets(base, scaled)))
    print(f"DOSE      {len(jobs)} practice x stimulation combinations ...", flush=True)
    dose_rows = run_plans(jobs, args.workers)
    plans = {j[0]: j[2] for j in jobs}
    dose_ranked = sorted(dose_rows, key=rank_key)
    best_label = dose_ranked[0]["plan"]
    best_plan = plans[best_label]

    # TIME
    days = [2, 7, 14, 30, 60] if args.quick else [2, 7, 14, 30, 45, 60, 90]
    print(f"TIME      starting the plan on days {days} ...", flush=True)
    time_jobs = [(f"start day {d}", "timing", variant(best_plan, therapy={"start_day": d}, stimulation={"start_day": d}))
                 for d in days]
    time_rows = run_plans(time_jobs, args.workers)

    # PREDICT
    std = standard_target(cfg, care)
    compare = [
        ("Recommended: " + best_label, "recommended", best_plan),
        ("Standard care: physical practice", "standard_care", variant(cfg, therapy={"mode": "physical"})),
        (f"Standard target: physical + {std['area']} {std['strength']:+}", "standard_target",
         with_targets(variant(cfg, therapy={"mode": "physical"}), [{"area": std["area"], "strength": std["strength"]}])),
        ("No therapy", "none", without_care(cfg)),
        ("Recommended, split brain (control)", "split", variant(best_plan, corpus_callosum={"strength": 0.0})),
    ]
    print("PREDICT   comparing with standard care ...", flush=True)
    pred_rows = run_plans(compare, args.workers)
    rec, care_row, std_row = pred_rows[0], pred_rows[1], pred_rows[2]

    target_map(target_rows, cfg, {k_: float(v) for k_, v in patient["damage"].items()}, out_dir / "1_where_to_help.png")
    recovery_curves({r["plan"].split(":")[0] if r["kind"] != "recommended" else "Recommended": r["weekly_function"]
                     for r in pred_rows}, cfg["care"]["goal"], f"{card['name']}: predicted recovery",
                    out_dir / "2_recovery.png")

    lines = [f"# Personal plan: {card['name']}\n",
             "*Model prediction from a two-hemisphere brain simulation (HCP connectome, Glasser areas). "
             "Not medical advice.*\n",
             "## 1. Identify\n",
             f"- Disorder: **{card['disorder']}**, {card['damaged_hemisphere']} hemisphere; epicentre **{card['epicentre']}**",
             "- Damaged areas: " + ", ".join(f"{a} ({d:.2f})" for a, d in list(card["damaged_areas"].items())[:12]),
             "- Jobs hit (mean damage of their areas): " + ", ".join(f"{j} {d:.2f}" for j, d in card["jobs_hit"].items()),
             f"- Job trained in therapy: **{card['trained_job']}**\n",
             "## 2. Recruit: where to help on the healthy side\n",
             f"Best: **{target_text(targets)}**. "
             f"Today's standard target for comparison: {std['area']} {std['strength']:+} ({std['why']}).\n",
             *rows_table(sorted(target_rows, key=rank_key), 10), "", "![where to help](1_where_to_help.png)\n",
             "## 3. Dose: which practice, how much stimulation\n", *rows_table(dose_ranked, 10), "",
             "## 4. Time: the golden period\n", *rows_table(time_rows), "",
             "## 5. Predict\n", *rows_table(pred_rows), "",
             f"**Recommended vs standard care:** {savings(rec, care_row)}.\n",
             f"**Recommended vs standard target:** {savings(rec, std_row)}.\n",
             "![recovery](2_recovery.png)\n",
             "## Assumptions\n",
             "- Recovered = the trained job averages at least "
             f"{cfg['care']['goal'] * 100:.0f}% of its healthy strength over two weeks in a row.",
             "- 1 model step = 1 hour; practice = 2 hours (300 repetitions) on 5 days a week; stimulation only during practice.",
             "- Imagery strength, VNS multiplier, BCI bonus, sleep replay and the golden-period curve are "
             "literature-based assumptions (docs/factors.md); results are testable predictions, not clinical proof."]
    (out_dir / "personal_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "plan.json").write_text(json.dumps({"card": card, "targets": targets, "recommended": rec,
                                                   "standard_care": care_row, "standard_target": std_row,
                                                   "timing": time_rows}, indent=1, default=float), encoding="utf-8")
    print("\n".join(rows_table(pred_rows)))
    print(f"\nRecommended vs standard care: {savings(rec, care_row)}")
    print(f"Saved {out_dir / 'personal_plan.md'}")


if __name__ == "__main__":
    sys.exit(main())
