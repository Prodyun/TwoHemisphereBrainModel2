"""Poster figures and a results summary from the latest batch (and the latest fMRI fit, if any).

    python scripts/make_figures.py                       # uses the newest results/batch_*/
    python scripts/make_figures.py --batch results/batch_20261122_101500

Statistics: each starting area is one independent case. Seeds are averaged within an area first,
then we report the mean over areas with a 95% bootstrap confidence interval (10,000 resamples)
and how many areas showed a benefit. Figures go to results/figures_<time>/.
"""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain_model.hcp import load_hcp_brain
from brain_model.simulation import run_experiment, rolling_function_score, summarize
from brain_model.utils import load_config

BLUE, GREY, ORANGE = "#2a78d6", "#9a9890", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.dpi": 100,
})
MAIN_CONDITIONS = [  # (experiment, label on the chart)
    ("main", "Left disorder, real callosum (main)"),
    ("right_lesion", "Right disorder"),
    ("stroke", "Stroke instead of spreading"),
    ("no_plasticity", "No plasticity"),
    ("mirror_callosum", "Mirror-only callosum"),
    ("both_callosum", "Real + mirror callosum"),
]


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    text_columns = {"experiment", "condition", "start_area", "damaged_hemisphere", "lesion_type", "wiring"}
    for r in rows:
        for k, v in r.items():
            if k in text_columns:
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def per_area(rows: list[dict], experiment: str, column: str) -> dict[str, float]:
    """Seed-averaged value of `column` for each start area."""
    acc = defaultdict(list)
    for r in rows:
        if r["experiment"] == experiment and r.get(column) not in ("", None):
            acc[r["start_area"]].append(r[column])
    return {a: float(np.mean(v)) for a, v in acc.items()}


def bootstrap_ci(values, n=10_000, seed=0) -> tuple[float, float, float]:
    v = np.asarray(list(values), dtype=float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v))).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stat_line(label: str, values: dict[str, float], unit_scale=100) -> str:
    m, lo, hi = bootstrap_ci(values.values())
    pos = sum(v > 0 for v in values.values())
    return (f"| {label} | {m * unit_scale:+.1f} | {lo * unit_scale:+.1f} to {hi * unit_scale:+.1f} | "
            f"{pos}/{len(values)} |")


def fig_benefit_by_experiment(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = []
    for i, (exp, label) in enumerate(MAIN_CONDITIONS):
        vals = per_area(rows, exp, "benefit")
        if not vals:
            continue
        m, lo, hi = bootstrap_ci(vals.values())
        y = len(labels)
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.array(list(vals.values())) * 100, y + jitter, s=14, color=BLUE, alpha=0.25, lw=0)
        ax.plot([lo * 100, hi * 100], [y, y], color=INK, lw=2, solid_capstyle="round")
        ax.scatter([m * 100], [y], s=60, color=BLUE, edgecolor="white", lw=2, zorder=3)
        ax.text(hi * 100 + 0.6, y, f"{m * 100:+.1f}", va="center", color=INK, fontsize=9)
        labels.append(label)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Help from the healthy hemisphere (percentage points of function)")
    fig.suptitle("How much the corpus callosum helps the damaged hemisphere", x=0.02, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Dot = mean over start areas, line = 95% CI, faint dots = individual start areas",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_functions(rows, out: Path) -> None:
    panels = [("main", "Real callosum"), ("mirror_callosum", "Mirror-only callosum")]
    funcs = [k.split(":", 1)[1] for k in rows[0] if k.startswith("benefit:")
             and any(r["experiment"] == "main" and r.get(k) not in ("", None) for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(8, 3.2), sharey=True)
    for ax, (exp, title) in zip(axes, panels):
        y = np.arange(len(funcs))
        con = [np.mean(list(per_area(rows, exp, f"connected:{f}").values())) * 100 for f in funcs]
        spl = [np.mean(list(per_area(rows, exp, f"split:{f}").values())) * 100 for f in funcs]
        ax.barh(y - 0.19, con, 0.36, color=BLUE, label="Connected brain")
        ax.barh(y + 0.19, spl, 0.36, color=GREY, label="Split brain (control)")
        for yi, c, s in zip(y, con, spl):
            ax.text(max(c, s) + 1.5, yi, f"{c - s:+.0f} pts", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [f.replace("_", " ") for f in funcs])
        ax.set_xlim(0, 118)
        ax.set_xlabel("Function kept (% of healthy)")
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()  # shared axis: invert once so the first ability is on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle("Which abilities of the damaged left hemisphere are rescued", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_time_course(cfg: dict, out: Path) -> dict:
    results = run_experiment(cfg)
    summary = summarize(results, cfg)
    side, window = cfg["lesion"]["hemisphere"], cfg["analysis"]["window"]
    t = np.arange(cfg["steps"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    damage = results["connected"]["run"][f"{side}_damage"].mean(axis=1) * 100
    ax.fill_between(t, 0, damage, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(t, damage, color=ORANGE, lw=1.2)
    ax.text(t[-1], damage[-1] + 3, "damage", color=INK2, ha="right", fontsize=9)
    ends = {}
    for name, color in (("split", GREY), ("connected", BLUE)):
        res = results[name]
        s = rolling_function_score(res["run"][side], res["healthy"][side], res["run"][f"{side}_task"], window, 1)
        s = np.array([np.nan if v is None else v * 100 for v in s])
        label = "connected brain" if name == "connected" else "split brain (control)"
        ax.plot(t, s, color=color, lw=2, label=label)
        ends[label] = s[-1]
    # end-of-line labels, pushed apart so they never overlap
    hi, lo = max(ends, key=ends.get), min(ends, key=ends.get)
    y_hi, y_lo = ends[hi], ends[lo]
    if y_hi - y_lo < 7:
        mid = (y_hi + y_lo) / 2
        y_hi, y_lo = mid + 3.5, mid - 3.5
    ax.text(t[-1] + 10, y_hi, f"{hi} {ends[hi]:.0f}%", color=INK, va="center", fontsize=9)
    ax.text(t[-1] + 10, y_lo, f"{lo} {ends[lo]:.0f}%", color=INK, va="center", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.02, 0.2))
    ax.axvline(cfg["lesion"]["start_step"], color=INK2, lw=0.8, ls=":")
    ax.text(cfg["lesion"]["start_step"] + 8, 8, "disorder starts", color=INK2, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time (simulation steps)")
    ax.set_ylabel("% (function vs healthy; damage)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    region = cfg["lesion"].get("region", "")
    ax.set_title(f"Left hemisphere over time, disorder starting at {region} (Broca's area)", loc="left")
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return summary


def fig_benefit_map(rows, cfg, out: Path) -> None:
    brain = load_hcp_brain(cfg["anatomy"]["file"])
    names = brain.side_names("left")
    s = brain.sides["left"]
    vals = per_area(rows, "main", "benefit")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    v = s["verts"]
    lateral = v[:, 0] < np.percentile(v[:, 0], 60)  # outer surface, seen from the left
    ax.scatter(-v[lateral, 1], v[lateral, 2], s=0.3, color="#d9d8d2", lw=0, rasterized=True)
    pts = np.array([s["anchors"][names.index("L_" + a)] for a in vals])
    b = np.array(list(vals.values())) * 100
    sc = ax.scatter(-pts[:, 1], pts[:, 2], c=b, cmap="Blues", vmin=0, vmax=max(b.max(), 1e-6),
                    s=110, edgecolor=INK2, lw=0.8, zorder=3)
    for (x, z), name in zip(pts[:, 1:], vals):
        ax.text(-x, z + 0.045, name, ha="center", fontsize=7.5, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("help from the right hemisphere (pts)", color=INK2)
    ax.set_title("Where the disorder starts matters (left hemisphere, front is left)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_cross_spread(rows, out: Path) -> None:
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["experiment"] == "cross_spread":
            acc[r["cross_spread"]]["benefit"].append(r["benefit"] * 100)
            acc[r["cross_spread"]]["healthy"].append(r["damage_healthy_side_connected"] * 100)
    if not acc:
        return
    xs = sorted(acc)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    for ax, key, color, title, ylabel in (
            (axes[0], "benefit", BLUE, "Help to the damaged side", "percentage points"),
            (axes[1], "healthy", ORANGE, "Damage leaking to the healthy side", "% of healthy side damaged")):
        m = [np.mean(acc[x][key]) for x in xs]
        sd = [np.std(acc[x][key]) for x in xs]
        ax.errorbar(xs, m, yerr=sd, color=color, lw=2, marker="o", ms=7, mec="white", mew=2, capsize=0)
        ax.set_title(title)
        ax.set_xlabel("cross_spread (how easily the disorder crosses)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    fig.suptitle("Trade-off: the bridge that helps can also spread the disorder", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def newest(pattern: str) -> Path | None:
    found = sorted(Path("results").glob(pattern))
    return found[-1] if found else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=None)
    parser.add_argument("--fit", default=None)
    parser.add_argument("--config", default="configs/hcp.yaml")
    parser.add_argument("--out", default=None, help="output folder (default: results/figures_<time>)")
    args = parser.parse_args()
    batch = Path(args.batch) if args.batch else newest("batch_*")
    if batch is None:
        raise SystemExit("No batch results. Run: python scripts/run_batch.py")
    rows = load_rows(batch / "results.csv")
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else Path("results") / datetime.now().strftime("figures_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    fig_benefit_by_experiment(rows, out / "1_callosum_benefit_by_experiment.png")
    fig_functions(rows, out / "2_functions_rescued.png")
    headline = fig_time_course(cfg, out / "3_time_course_broca.png")
    fig_benefit_map(rows, cfg, out / "4_benefit_by_start_area.png")
    fig_cross_spread(rows, out / "5_cross_spread_tradeoff.png")

    lines = [f"# Results summary\n\nBatch: `{batch}`. Each start area is one case (seeds averaged).\n",
             "## Help from the healthy hemisphere (connected minus split brain, percentage points)\n",
             "| Experiment | Mean | 95% CI | Areas with benefit |", "|---|---|---|---|"]
    for exp, label in MAIN_CONDITIONS:
        vals = per_area(rows, exp, "benefit")
        if vals:
            lines.append(stat_line(label, vals))
    lines += ["\n## Paired comparisons (same start areas; difference in help, points)\n",
              "| Comparison | Mean difference | 95% CI | Areas where first > second |", "|---|---|---|---|"]
    main = per_area(rows, "main", "benefit")
    for exp, label in (("mirror_callosum", "mirror-only minus real callosum"),
                       ("both_callosum", "real+mirror minus real callosum"),
                       ("no_plasticity", "no plasticity minus plasticity"),
                       ("right_lesion", "right disorder minus left disorder")):
        other = per_area(rows, exp, "benefit")
        common = sorted(set(main) & set(other))
        if common:
            lines.append(stat_line(label, {a: other[a] - main[a] for a in common}))
    lines += ["\n## Per ability, main experiment (points)\n", "| Ability | Mean help | 95% CI | Areas with benefit |",
              "|---|---|---|---|"]
    for k in rows[0]:
        if k.startswith("benefit:"):
            vals = per_area(rows, "main", k)
            if vals:
                lines.append(stat_line(k.split(":", 1)[1].replace("_", " "), vals))
    lines += [f"\n## Headline run (start at {cfg['lesion'].get('region')})\n",
              f"- damage: {headline['connected']['final_mean_damage']:.0%} of the left hemisphere",
              f"- function kept: connected {headline['connected']['damaged_side_score']:.0%}, "
              f"split {headline['split']['damaged_side_score']:.0%} "
              f"({headline['benefit_of_callosum'] * 100:+.1f} points)"]
    fit_dir = Path(args.fit) if args.fit else newest("fit_*")
    if fit_dir and (fit_dir / "fit.json").exists():
        fit = json.loads((fit_dir / "fit.json").read_text(encoding="utf-8"))
        lines += [f"\n## Fit to real resting-state fMRI (`{fit_dir}`)\n",
                  "| Model | Target | Score | Left | Right | Between |", "|---|---|---|---|---|---|"]
        for model, per_target in fit["models"].items():
            for t, r in per_target.items():
                lines.append(f"| {model} | {t} | {r['score']:.3f} | {r['r_left']:.2f} | "
                             f"{r['r_right']:.2f} | {r['r_between']:.2f} |")
        for t, r in fit["wiring_alone"].items():
            lines.append(f"| wiring alone (reference) | {t} | {r['score']:.3f} | {r['r_left']:.2f} | "
                         f"{r['r_right']:.2f} | {r['r_between']:.2f} |")
        if fit.get("held_out"):
            lines += ["\nHeld-out check (settings fitted to the group, re-run with new random noise):\n",
                      "| Model | Target | Score |", "|---|---|---|"]
            for model, per_target in fit["held_out"].items():
                for t, r in per_target.items():
                    lines.append(f"| {model} | {t} | {r['score']:.3f} |")
        for png in fit_dir.glob("*.png"):
            shutil.copy(png, out / f"6_{png.name}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved figures to {out}")


if __name__ == "__main__":
    main()
