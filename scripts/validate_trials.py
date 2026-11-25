"""R-EXP-10: does the model reproduce real stroke trials? Virtual cohorts, honest comparison.

    python scripts/validate_trials.py            # 24 virtual motor-stroke patients, ~2-4 minutes
    python scripts/validate_trials.py --quick    # 8 patients (a check)

Cohort: strokes in the motor areas of either side (4, 6d, 3b, 6a, 1, 6mp; left and right) at two
severities, each damaging its 4 most strongly wired neighbours too ("mixed lesion location and extent",
as in E-FIT). Everyone gets the same physical practice (care.yaml). Three trials:

  NICHE / E-FIT (INDEPENDENT test: nothing in the model was tuned to it)
      practice vs practice + calming the healthy M1 (1 Hz rTMS, strength -0.3).
      Trial: not better than sham (E-FIT, Stroke 2023; NICHE, Harvey 2018).
      Plus the personalised Help-or-Hinder switch: excite, calm or leave the healthy side per patient.
  VNS-REHAB (CALIBRATION check: the VNS multiplier 2.1 was taken from this trial)
      practice vs practice + paired VNS. Trial: clinically meaningful response 47% vs 24%, about 2x
      (Dawson 2021 Lancet). Model "response" = reaching the goal (80%).
  CPASS (CALIBRATION check: the golden-period curve was taken from this trial)
      4 weeks of extra practice started early (day 15), subacute (day 75) or chronic (day 180), vs none.
      Trial: subacute > early > chronic (+6.87, +5.25, +2.41 ARAT; Dromerick 2021 PNAS).

Writes results/validation/report.md, results.json and validation.png. Simulation outputs only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from brain_model.patient import patient_config, patient_from_area  # noqa: E402
from brain_model.planning import run_plans, standard_target, variant, with_targets  # noqa: E402
from brain_model.healthy_side import better, meaningful  # noqa: E402
from brain_model.utils import load_config  # noqa: E402

AREAS = ["4", "6d", "3b", "6a", "1", "6mp"]


def cohort(base, care, quick=False):
    areas = AREAS[:2] if quick else AREAS
    out = []
    for side in ("L_", "R_"):
        for a in areas:
            for sev in (0.6, 0.9):
                p = patient_from_area(side + a, sev, "stroke")
                p["name"] = f"{side}{a} s{sev}"
                cfg = patient_config(base, care, p)
                cfg["therapy"]["function"] = "movement"     # all trials score the arm (movement job)
                out.append(cfg)
    return out


def responders(rows):
    return float(np.mean([r["weeks_to_goal"] is not None for r in rows]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="results/validation")
    args = ap.parse_args()
    os.chdir(ROOT)
    base, care = load_config("configs/hcp.yaml"), load_config("configs/care.yaml")
    pts = cohort(base, care, args.quick)
    print(f"{len(pts)} virtual patients", flush=True)

    # ---- NICHE / E-FIT + Help-or-Hinder ----
    jobs = []
    for i, cfg in enumerate(pts):
        std = standard_target(cfg, care)          # the healthy M1 for the movement job
        jobs += [(f"{i}|practice", "practice", cfg),
                 (f"{i}|calm", "calm", with_targets(cfg, [{"area": std["area"], "strength": -0.3}])),
                 (f"{i}|excite", "excite", with_targets(cfg, [{"area": std["area"], "strength": 0.5}])),
                 (f"{i}|vns", "vns", variant(cfg, therapy={"mode": ["physical", "vns"]}))]
    rows = run_plans(jobs, args.workers)
    by = {}
    for r in rows:
        i, kind = r["plan"].split("|")
        by.setdefault(int(i), {})[kind] = r
    tol = meaningful(pts[0])
    diff_calm = [by[i]["calm"]["final_function"] - by[i]["practice"]["final_function"] for i in by]
    switch, words = [], []
    for i in by:
        best = by[i]["practice"]
        word = "LEAVE"
        for kind, w in (("excite", "EXCITE"), ("calm", "CALM")):
            if better(by[i][kind], best, tol):
                best, word = by[i][kind], w
        switch.append(best["final_function"] - by[i]["practice"]["final_function"])
        words.append(word)
    niche = {"calm_everyone_mean_points": round(100 * float(np.mean(diff_calm)), 2),
             "calm_everyone_sd_points": round(100 * float(np.std(diff_calm)), 2),
             "responders_practice": responders([by[i]["practice"] for i in by]),
             "responders_calm": responders([by[i]["calm"] for i in by]),
             "personalised_mean_points": round(100 * float(np.mean(switch)), 2),
             "switch_words": {w: words.count(w) for w in ("EXCITE", "CALM", "LEAVE")}}
    niche["agrees_with_trial"] = abs(niche["calm_everyone_mean_points"]) < 2.0 and \
        abs(niche["responders_calm"] - niche["responders_practice"]) <= 0.1

    # ---- VNS-REHAB ----
    r_p, r_v = responders([by[i]["practice"] for i in by]), responders([by[i]["vns"] for i in by])
    vns = {"responders_practice": r_p, "responders_vns": r_v,
           "ratio": round(r_v / r_p, 2) if r_p > 0 else None,
           "mean_gain_points": round(100 * float(np.mean([by[i]["vns"]["final_function"] - by[i]["practice"]["final_function"] for i in by])), 2),
           "trial": "47% vs 24% responders (about 2x)"}
    vns["agrees_with_trial"] = r_v > r_p

    # ---- CPASS ----
    starts = {"early (day 15)": 15, "subacute (day 75)": 75, "chronic (day 180)": 180}
    # CPASS enrolled patients with lasting moderate-severe arm impairment: use the severe virtual patients
    sub = [c for c in pts if c["patient"]["name"].endswith("s0.9")]
    jobs = []
    for i, cfg in enumerate(sub):
        for label, d in starts.items():
            plan = variant(cfg, care={"days": d + 28 + 14}, therapy={"start_day": d, "end_day": d + 28})
            plan["steps"] = int((cfg["care"]["onset_day"] + d + 42) * cfg["time"]["steps_per_day"])
            jobs.append((f"{i}|{label}|extra", "cpass", plan))
            jobs.append((f"{i}|{label}|none", "cpass", variant(plan, therapy={"mode": "none"})))
    rows = run_plans(jobs, args.workers)
    res = {}
    for r in rows:
        i, label, arm = r["plan"].split("|")
        res.setdefault(label, {}).setdefault(i, {})[arm] = r["final_function"]
    cpass = {label: round(100 * float(np.mean([v["extra"] - v["none"] for v in d.values()])), 2) for label, d in res.items()}
    order = sorted(cpass, key=cpass.get, reverse=True)
    cpass_out = {"gain_points": cpass, "order": order, "trial_order": list(starts)[1::-1] + ["chronic (day 180)"]}
    cpass_out["agrees_with_trial"] = order[0].startswith("subacute") and order[-1].startswith("chronic")

    out = {"n_patients": len(pts), "niche_efit": niche, "vns_rehab": vns, "cpass": cpass_out}
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(out, indent=2))
    write_report(out, od)
    figure(out, od / "validation.png")
    print(json.dumps(out, indent=2))
    return 0


def write_report(o: dict, od: Path) -> None:
    n, v, c = o["niche_efit"], o["vns_rehab"], o["cpass"]
    ok = lambda b: "✔ agrees" if b else "✖ does not agree"   # noqa: E731
    lines = [
        "# Does the model reproduce real stroke trials? (R-EXP-10)", "",
        f"{o['n_patients']} virtual motor-stroke patients (mixed areas, both sides, two severities). Simulation only.", "",
        "| Trial | Real result | Model | Verdict | Kind of test |", "|---|---|---|---|---|",
        f"| NICHE / E-FIT: calm the healthy M1 | not better than sham | {n['calm_everyone_mean_points']:+.1f} points "
        f"(SD {n['calm_everyone_sd_points']:.1f}); responders {n['responders_calm']:.0%} vs {n['responders_practice']:.0%} | "
        f"{ok(n['agrees_with_trial'])} | independent |",
        f"| VNS-REHAB: paired VNS | 47% vs 24% responders | {v['responders_vns']:.0%} vs {v['responders_practice']:.0%} "
        f"(x{v['ratio']}); final function {v['mean_gain_points']:+.1f} points | "
        f"{'✔ agrees' if v['agrees_with_trial'] else ('½ same direction in function, not in responders' if v['mean_gain_points'] > 0 else '✖ does not agree')} | "
        f"calibration (VNS factor taken from this trial) |",
        f"| CPASS: when to add practice | subacute > early > chronic | " +
        ", ".join(f"{k} {val:+.1f}" for k, val in c["gain_points"].items()) +
        f" | {ok(c['agrees_with_trial'])} | calibration (golden period taken from this trial) |", "",
        "## The personalised switch (Help-or-Hinder)",
        f"Choosing EXCITE / CALM / LEAVE per patient: {n['personalised_mean_points']:+.1f} points on average, versus "
        f"{n['calm_everyone_mean_points']:+.1f} for calming everyone. Choices: {n['switch_words']}.", "",
        "## Honest notes",
        "- Only NICHE / E-FIT is an independent test; VNS and CPASS numbers were used to set the model, so matching",
        "  them only shows the settings were carried through correctly.",
        "- The model's interhemispheric brake is off by default, so calming the healthy side cannot help in the model;",
        "  the NICHE / E-FIT match is therefore partly built in. Turning the brake on (corpus_callosum.inhibition) is a",
        "  sensitivity test still to run.",
        "- 'Responder' = reaching 80% of healthy function in the model, not a Fugl-Meyer change; when most virtual",
        "  patients recover anyway (a ceiling), responder rates cannot show a difference. VNS-REHAB enrolled chronic",
        "  patients (9 months to 10 years after stroke); this cohort is treated from day 7.",
        "- CPASS uses only the severe virtual patients (the trial enrolled lasting moderate-severe impairment).",
        "- A mismatch on a calibration check is a finding: it means the rest of the model (spontaneous healing, the",
        "  permanent-damage floor, plasticity) changes what the borrowed number alone would predict.",
    ]
    (od / "report.md").write_text("\n".join(lines), encoding="utf-8")


def figure(o: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    n = o["niche_efit"]
    ax[0].bar(["practice", "+ calm\nhealthy M1", "personalised\nswitch"],
              [0, n["calm_everyone_mean_points"], n["personalised_mean_points"]], color=["#999", "#d9534f", "#2a9d8f"])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("NICHE / E-FIT (independent)")
    ax[0].set_ylabel("change in final function (points)")
    v = o["vns_rehab"]
    ax[1].bar(["practice", "+ paired VNS"], [100 * v["responders_practice"], 100 * v["responders_vns"]], color=["#999", "#2a78d6"])
    ax[1].set_title("VNS-REHAB (calibration)")
    ax[1].set_ylabel("responders (%)")
    c = o["cpass"]["gain_points"]
    ax[2].bar([k.split(" (")[0] for k in c], list(c.values()), color="#e9a03b")
    ax[2].set_title("CPASS timing (calibration)")
    ax[2].set_ylabel("gain from 4 weeks practice (points)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
