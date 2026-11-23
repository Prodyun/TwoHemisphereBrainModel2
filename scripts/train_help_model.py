"""Train and test the AI Help Predictor on the simulated patients.

    python scripts/build_help_dataset.py      # first (the training data)
    python scripts/train_help_model.py

Saves the model to data/processed/help_model.joblib and a report with figures to
results/help_model/: scores on areas the model never saw, comparison with baselines, the most
important area features, and a brain map of "chance the healthy side can help" for every area.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from brain_model.help_model import MODEL_FILE, HelpPredictor, area_features, train  # noqa: E402
from brain_model.hcp import load_hcp_brain  # noqa: E402
from brain_model.visualization.plan_figures import GAIN_CMAP, _surface  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"


def chance_map(pred: HelpPredictor, out: Path) -> dict:
    b = load_hcp_brain()
    chance = {}
    for nm in b.names:
        res = pred.predict(nm, 0.9, "stroke")
        chance[nm] = {"chance": max(m["chance"] for m in res["modes"]),
                      "help_points": res["modes"][0]["help_points"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, side in zip(axes, ("left", "right")):
        vals = np.array([chance[nm]["help_points"] for nm in b.side_names(side)])
        sc = _surface(ax, b.sides[side], vals, GAIN_CMAP, TwoSlopeNorm(vcenter=0, vmin=-10, vmax=20),
                      f"{side.capitalize()} hemisphere damaged")
    fig.colorbar(sc, ax=axes, shrink=0.7).set_label("help from the other side\n(points of function vs split brain, best mode)",
                                                   color=INK2)
    fig.suptitle("AI Help Predictor: which areas can the healthy hemisphere help? (severe stroke)", x=0.02,
                 ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return chance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/help_dataset/rows.jsonl")
    ap.add_argument("--out", default="results/help_model")
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    if errors:
        print(f"Skipping {len(errors)} failed cases, e.g. {errors[0]['area']}: {errors[0]['error']}")
    feats = area_features()
    print(f"Training on {len(rows)} simulated patients, {len(feats.columns)} area features ...", flush=True)
    bundle = train(rows, feats)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: v for k, v in bundle.items() if k != "oof"}, MODEL_FILE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = bundle["metrics"]
    oof = bundle["oof"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(oof["y_help"], oof["help"], s=4, alpha=0.4, color="#2a78d6")
    lim = [min(oof["y_help"].min(), oof["help"].min()), max(oof["y_help"].max(), oof["help"].max())]
    axes[0].plot(lim, lim, color=INK2, lw=1)
    axes[0].set_xlabel("help in the simulation (points)")
    axes[0].set_ylabel("AI prediction (points)")
    axes[0].set_title(f"Areas the AI never saw: R² = {m['r2_help']:.2f}", loc="left", fontsize=10)
    top = bundle["importance"][:12][::-1]
    axes[1].barh([t[0] for t in top], [t[1] for t in top], color="#1f9e6e")
    axes[1].set_title("What matters most (permutation importance)", loc="left", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "1_model_check.png", dpi=180)
    plt.close(fig)

    pred = HelpPredictor({k: v for k, v in bundle.items() if k != "oof"}, feats)
    chance = chance_map(pred, out / "2_chance_map.png")
    ranked = sorted(chance.items(), key=lambda kv: -kv[1]["help_points"])
    lines = ["# AI Help Predictor: model report\n",
             f"Trained on {m['rows']} simulated patients ({m['areas']} damaged areas). "
             f"Scores are for **areas the model never saw** (5-fold cross-validation grouped by area).\n",
             "| Measure | AI model | Baseline |", "|---|---|---|",
             f"| Chance of reaching the goal: AUC | {m['auc_success']:.3f} | {m['auc_success_wiring_only']:.3f} (wiring only) |"
             if m["auc_success"] is not None else "| Chance of reaching the goal | n/a | n/a |",
             f"| Help from the other side: R² | {m['r2_help']:.3f} | {m['r2_help_wiring_only']:.3f} (wiring only) |",
             f"| Help: mean error (points) | {m['mae_help_points']:.2f} | {m['mae_help_always_average']:.2f} (always the average) |",
             f"| Final function: R² | {m['r2_final']:.3f} | |", "",
             "## Most important features\n", *[f"- {k}: {v:.3f}" for k, v in bundle["importance"][:15]], "",
             "## Areas the healthy side helps most and least (severe stroke, best mode, AI prediction)\n",
             "Most: " + ", ".join(f"{a} {c['help_points']:+.1f} pts ({c['chance'] * 100:.0f}% reach the goal)"
                                  for a, c in ranked[:10]),
             "", "Least: " + ", ".join(f"{a} {c['help_points']:+.1f} pts ({c['chance'] * 100:.0f}% reach the goal)"
                                       for a, c in ranked[-10:]), "",
             "## Caveat\n",
             "`right_side` ranks high mainly because the model gives each hemisphere's vision job a different "
             "rhythm (each eye sees a different scene), so signals from the other side interfere with the right "
             "visual areas (R_V1-V4 get negative help). This follows from the model's assumptions; it is not a finding.",
             "",
             "![model check](1_model_check.png)", "", "![chance map](2_chance_map.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "chance_by_area.json").write_text(json.dumps(chance, indent=0), encoding="utf-8")
    print("\n".join(lines[2:9]))
    print(f"Saved {MODEL_FILE} and {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
