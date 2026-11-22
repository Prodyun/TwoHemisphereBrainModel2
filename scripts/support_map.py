"""Does the healthy hemisphere help, wherever the disorder starts? Every one of the 360 areas.

    python scripts/support_map.py                 # 360 start areas x 3 seeds, about 20 minutes
    python scripts/support_map.py --seeds 1       # quick version

For each start area (180 in the left hemisphere, 180 in the right) the disorder is run in a
connected brain and in a split brain with the same noise. "Help" = function kept by the damaged
hemisphere when connected minus when split (percentage points). Negative help means the signals
from the other side got in the way (interference). Nothing is tuned to make help positive.

Output (results/support_map/): support.csv (one row per start area, seeds averaged),
7_support_map.png (both hemispheres coloured by help) and support_summary.md.
Finished runs are saved at once, so an interrupted run continues where it stopped.
"""
import argparse
import copy
import csv
import importlib.util
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from brain_model.hcp import load_hcp_brain  # noqa: E402
from brain_model.utils import load_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_batch", Path(__file__).with_name("run_batch.py"))
run_batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_batch)

INK, INK2 = "#0b0b0b", "#52514e"
HELP_CMAP = LinearSegmentedColormap.from_list("help", ["#c8412f", "#d9d8d2", "#2a78d6"])  # interferes / none / helps


def jobs(base: dict, names: dict[str, list[str]], seeds: list[int]) -> list[tuple]:
    out = []
    for side in ("left", "right"):
        for name in names[side]:
            for seed in seeds:
                cfg = copy.deepcopy(base)
                cfg["seed"] = seed
                cfg["lesion"].update(hemisphere=side, region=name)
                out.append((f"support_{side}", f"{side} start", name[2:], cfg))
    return out


def per_start(rows: list[dict]) -> list[dict]:
    """Average the seeds of each start area."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["damaged_hemisphere"], r["start_area"])].append(r)
    out = []
    for (side, area), rs in sorted(groups.items()):
        row = {"damaged_hemisphere": side, "start_area": area, "seeds": len(rs)}
        for key in rs[0]:
            if key == "benefit" or key.startswith("benefit:") or key.startswith("score_"):
                row[key] = float(np.mean([float(r[key]) for r in rs]))
        out.append(row)
    return out


def bootstrap_ci(values, n=10_000, seed=0):
    v = np.asarray(values, float)
    means = np.random.default_rng(seed).choice(v, (n, len(v))).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def figure(starts: list[dict], cfg: dict, out: Path) -> None:
    brain = load_hcp_brain(cfg["anatomy"]["file"])
    vals = {s: np.full(brain.n, np.nan) for s in ("left", "right")}
    for r in starts:
        side = r["damaged_hemisphere"]
        names = brain.side_names(side)
        vals[side][names.index(("L_" if side == "left" else "R_") + r["start_area"])] = r["benefit"] * 100
    lim = max(1.0, np.nanpercentile(np.abs(np.concatenate(list(vals.values()))), 98))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    for row, side in enumerate(("left", "right")):
        s = brain.sides[side]
        v, reg = s["verts"], s["vertex_region"].astype(int)
        outer = v[:, 0] < np.median(v[:, 0]) if side == "left" else v[:, 0] > np.median(v[:, 0])
        for col, (mask, view) in enumerate(((outer, "outer side"), (~outer, "inner side"))):
            ax = axes[row, col]
            x = -v[mask, 1]  # front of the brain always on the left
            r = reg[mask]
            colour = np.where(r >= 0, vals[side][np.clip(r, 0, None)], np.nan)
            wall = np.isnan(colour)  # medial wall (no area)
            ax.scatter(x[wall], v[mask, 2][wall], s=1.2, color="#5a5a5a", lw=0, rasterized=True)
            sc = ax.scatter(x[~wall], v[mask, 2][~wall], c=colour[~wall], cmap=HELP_CMAP, norm=norm, s=1.2,
                            lw=0, rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{side.capitalize()} hemisphere damaged · {view}", fontsize=9.5, color=INK)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, location="right")
    cb.set_label("help from the other hemisphere (points)\nred = gets in the way · grey = none · blue = helps", color=INK2)
    fig.suptitle("Where the disorder starts decides how much the healthy hemisphere helps (front of brain on the left)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summary(starts: list[dict]) -> str:
    lines = ["# Support map: every start area\n",
             "Help = function kept by the damaged hemisphere, connected minus split brain (points). "
             "Each start area is one case (seeds averaged).\n",
             "| Damaged hemisphere | Areas | Mean help | 95% CI | Areas helped (> +0.5) | No real effect | Areas hurt (< -0.5) |",
             "|---|---|---|---|---|---|---|"]
    for side in ("left", "right"):
        b = np.array([r["benefit"] * 100 for r in starts if r["damaged_hemisphere"] == side])
        if not len(b):
            continue
        m, lo, hi = bootstrap_ci(b)
        lines.append(f"| {side} | {len(b)} | {m:+.1f} | {lo:+.1f} to {hi:+.1f} | {(b > 0.5).sum()} | "
                     f"{((b >= -0.5) & (b <= 0.5)).sum()} | {(b < -0.5).sum()} |")
    for side in ("left", "right"):
        rs = [r for r in starts if r["damaged_hemisphere"] == side]
        if not rs:
            continue
        lines += [f"\n## {side.capitalize()} hemisphere damaged: help per ability (mean over start areas, points)\n",
                  "| Ability | Mean help | Areas where it helps | Areas where it hurts |", "|---|---|---|---|"]
        for key in sorted(k for k in rs[0] if k.startswith("benefit:")):
            v = np.array([r[key] * 100 for r in rs])
            lines.append(f"| {key.split(':', 1)[1].replace('_', ' ')} | {v.mean():+.1f} | {(v > 0.5).sum()} | {(v < -0.5).sum()} |")
        order = sorted(rs, key=lambda r: r["benefit"])
        fmt = lambda r: f"{r['start_area']} ({r['benefit'] * 100:+.1f})"  # noqa: E731
        lines += [f"\nMost help: {', '.join(fmt(r) for r in order[::-1][:5])}",
                  f"\nLeast help: {', '.join(fmt(r) for r in order[:5])}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hcp.yaml")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", default="results/support_map")
    args = parser.parse_args()

    base = load_config(args.config)
    brain = load_hcp_brain(base["anatomy"]["file"])
    names = {s: brain.side_names(s) for s in ("left", "right")}
    todo = jobs(base, names, list(range(1, args.seeds + 1)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "results_partial.jsonl"
    rows = run_batch.load_partial(partial)
    done = {run_batch.job_key(r) for r in rows}
    remaining = [j for j in todo if run_batch.job_key(j) not in done]
    print(f"{len(todo)} runs ({len(todo) - len(remaining)} already done) on {args.workers} cores ...", flush=True)
    if remaining:
        with Pool(args.workers) as pool, open(partial, "a", encoding="utf-8") as log:
            for i, row in enumerate(pool.imap_unordered(run_batch.run_one, remaining, chunksize=1), 1):
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                os.fsync(log.fileno())
                if i % 50 == 0 or i == len(remaining):
                    print(f"  {i}/{len(remaining)}", flush=True)

    starts = per_start(rows)
    fields = ["damaged_hemisphere", "start_area", "seeds", "benefit"]
    fields += sorted({k for r in starts for k in r} - set(fields))
    with open(out_dir / "support.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(starts)
    figure(starts, base, out_dir / "7_support_map.png")
    text = summary(starts)
    (out_dir / "support_summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
