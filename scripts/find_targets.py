"""RECRUIT: which areas of the HEALTHY hemisphere should get extra stimulation for this patient?

    python scripts/find_targets.py --patient configs/patients/left_6d_stroke.yaml
    python scripts/find_targets.py --patient <file> --k 30 --strengths 0.5 -0.3

The healthy-side areas best placed to reach the damaged areas (real callosal fibres + one step
through the healthy side's own wiring) are each stimulated during the patient's practice, excited
and calmed, and ranked by the three outcomes. Always compared with: practice alone, today's
standard target, random targets and the split brain (control).
Output: results/plans/<patient>/targets.csv, targets.md, 1_where_to_help.png
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from brain_model.patient import load_patient, patient_config  # noqa: E402
from brain_model.planning import rank_key, search_targets, standard_target  # noqa: E402
from brain_model.utils import load_config  # noqa: E402
from brain_model.visualization.plan_figures import target_map  # noqa: E402

FIELDS = ["plan", "kind", "therapy", "targets", "final_function", "weeks_to_goal", "physical_reps_to_goal",
          "mental_reps_to_goal", "physical_reps_total", "mental_reps_total", "final_damage_trained_areas",
          "start_day", "callosum"]


def fmt(v, kind="f"):
    if v is None:
        return "not reached"
    return f"{v:,.0f}" if kind == "n" else f"{v * 100:.1f}%"


def table(rows: list[dict], top: int | None = None) -> list[str]:
    lines = ["| Plan | Final function | Weeks to goal | Physical reps to goal |", "|---|---|---|---|"]
    for r in (rows[:top] if top else rows):
        weeks = r["weeks_to_goal"] if r["weeks_to_goal"] is not None else "not reached"
        lines.append(f"| {r['plan']} | {fmt(r['final_function'])} | {weeks} | {fmt(r['physical_reps_to_goal'], 'n')} |")
    return lines


def save(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--config", default="configs/hcp.yaml")
    ap.add_argument("--care", default="configs/care.yaml")
    ap.add_argument("--k", type=int, default=20, help="how many pre-screened healthy-side areas to simulate")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.5, -0.3])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    care = load_config(args.care)
    patient = load_patient(args.patient)
    cfg = patient_config(load_config(args.config), care, patient)
    out_dir = Path(args.out or f"results/plans/{Path(args.patient).stem}")
    print(f"{patient['name']}: trained job '{cfg['therapy']['function']}', practice {cfg['therapy']['mode']}")
    print(f"Simulating {args.k} healthy-side areas x {len(args.strengths)} strengths on {args.workers} cores ...",
          flush=True)
    rows = search_targets(cfg, care, args.workers, args.k, tuple(args.strengths))
    save(rows, out_dir)
    std = standard_target(cfg, care)
    ranked = sorted(rows, key=rank_key)
    lines = [f"# Where to help: {patient['name']}\n",
             f"Trained job: **{cfg['therapy']['function']}** · practice: {cfg['therapy']['mode']} · "
             f"standard target for comparison: {std['area']} ({std['why']}).\n",
             "Stimulation is given only during practice. Function is scored only outside practice, so the "
             "gains below are lasting repair, not the stimulation itself.\n",
             "## All plans, best first\n", *table(ranked), "",
             "Strength +0.5 = 50% more excitable; -0.3 = 30% calmer. Model predictions, not clinical advice."]
    (out_dir / "targets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    target_map(rows, cfg, {k: float(v) for k, v in patient["damage"].items()}, out_dir / "1_where_to_help.png")
    print("\n".join(table(ranked, 12)))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
