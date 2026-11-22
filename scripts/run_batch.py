"""Run every experiment many times: many seeds x many starting areas. One CSV row per run.

    python scripts/run_batch.py                  # about 10 minutes, uses all CPU cores
    python scripts/run_batch.py --seeds 3        # quick version

Why many runs? One simulation could be a lucky or unlucky case. Repeating with different random
noise (seeds) and different starting areas shows whether the result holds in general.
Starting areas: Broca's area (44) plus areas spread evenly over the hemisphere.

Experiments (each run always compares the connected brain with the split brain, same noise):
  main            disorder spreads from each start area in the LEFT hemisphere (real callosum)
  right_lesion    the same, starting in the matching RIGHT-hemisphere area
  stroke          sudden damage that does not spread, instead of a spreading disorder
  no_plasticity   the brain does not adapt to damage (plasticity_rate = 0)
  mirror_callosum callosum links only matching areas (area i <-> area i)
  both_callosum   real fibres plus matching-area links
  cross_spread    the disorder can also leak through the callosum (0, 0.25, 0.5, 1), from area 44
"""
import argparse
import copy
import csv
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per worker; workers run side by side
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from brain_model.hcp import load_hcp_brain  # noqa: E402
from brain_model.simulation import run_experiment, summarize  # noqa: E402
from brain_model.utils import load_config  # noqa: E402


def spread_out_areas(centers: np.ndarray, first: int, k: int) -> list[int]:
    """`first` plus k-1 more areas, each as far as possible from those already chosen."""
    chosen = [first]
    dist = np.linalg.norm(centers - centers[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(centers - centers[nxt], axis=1))
    return chosen


def jobs(base: dict, areas: list[str], seeds: list[int]) -> list[tuple[str, str, str, dict]]:
    out = []

    def add(experiment, condition, area, changes):
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            for section, values in changes.items():
                cfg[section].update(values)
            out.append((experiment, condition, area, cfg))

    for area in areas:
        add("main", "left spreading", area, {"lesion": {"region": f"L_{area}"}})
        add("right_lesion", "right spreading", area,
            {"lesion": {"hemisphere": "right", "region": f"R_{area}"}})
        add("stroke", "left stroke", area, {"lesion": {"region": f"L_{area}", "type": "stroke"}})
        add("no_plasticity", "no plasticity", area,
            {"lesion": {"region": f"L_{area}"}, "corpus_callosum": {"plasticity_rate": 0.0}})
        add("mirror_callosum", "mirror callosum", area,
            {"lesion": {"region": f"L_{area}"}, "corpus_callosum": {"wiring": "mirror"}})
        add("both_callosum", "real + mirror callosum", area,
            {"lesion": {"region": f"L_{area}"}, "corpus_callosum": {"wiring": "both"}})
    for cs in (0.0, 0.25, 0.5, 1.0):
        add("cross_spread", f"cross_spread {cs}", "44", {"lesion": {"region": "L_44", "cross_spread": cs}})
    return out


def run_one(job: tuple[str, str, str, dict]) -> dict:
    experiment, condition, area, cfg = job
    summary = summarize(run_experiment(cfg), cfg)
    damaged = cfg["lesion"]["hemisphere"]
    con, spl = summary["connected"], summary["split"]
    row = {
        "experiment": experiment, "condition": condition, "start_area": area, "seed": cfg["seed"],
        "damaged_hemisphere": damaged, "lesion_type": cfg["lesion"].get("type", "spreading"),
        "wiring": cfg["corpus_callosum"].get("wiring", "mirror"),
        "plasticity_rate": cfg["corpus_callosum"]["plasticity_rate"],
        "cross_spread": cfg["lesion"].get("cross_spread", 0.0),
        "damage_damaged_side": con["final_mean_damage"],
        "damage_healthy_side_connected": con["healthy_side_final_damage"],
        "score_connected": con["damaged_side_score"], "score_split": spl["damaged_side_score"],
        "healthy_side_score_connected": con["healthy_side_score"],
        "benefit": summary["benefit_of_callosum"],
    }
    for name, benefit in summary["function_benefit"].items():
        row[f"benefit:{name}"] = benefit
        row[f"connected:{name}"] = con["functions"][damaged][name]
        row[f"split:{name}"] = spl["functions"][damaged][name]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hcp.yaml")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--areas", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    base = load_config(args.config)
    brain = load_hcp_brain(base["anatomy"]["file"])
    names = brain.side_names("left")
    chosen = spread_out_areas(brain.sides["left"]["centers"], names.index("L_44"), args.areas)
    areas = [names[i][2:] for i in chosen]
    print("Start areas:", ", ".join(areas))
    todo = jobs(base, areas, list(range(1, args.seeds + 1)))
    print(f"{len(todo)} experiments on {args.workers} cores ...")

    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, todo, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    out_dir = Path("results") / datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (":" in k, k))
    first = ["experiment", "condition", "start_area", "seed"]
    fields = first + [f for f in fields if f not in first]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["experiment"], r["condition"], r["start_area"], r["seed"])))
    print(f"Saved {out_dir / 'results.csv'}")
    print("Next: python scripts/make_figures.py")


if __name__ == "__main__":
    main()
