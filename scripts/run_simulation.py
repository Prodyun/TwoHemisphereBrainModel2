"""Entry point:  python scripts/run_simulation.py --config configs/default.yaml

Runs the lesion experiment (connected brain vs split brain), prints the result in plain
language, and saves to results/<timestamp>/:
  summary.json   the key numbers
  viewer.html    the interactive 3D viewer (double-click to open in a browser)
"""
import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

from brain_model.simulation import run_experiment, summarize
from brain_model.utils import load_config
from brain_model.visualization import build_viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-open", action="store_true", help="don't open the viewer automatically")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_experiment(config)
    summary = summarize(results, config)

    out_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    viewer = build_viewer(config, results, summary, out_dir / "viewer.html")

    side = summary["damaged_hemisphere"]
    con, spl = summary["connected"], summary["split"]
    print(f"A disorder started in the {side} hemisphere and spread to "
          f"{con['final_mean_damage']:.0%} average damage.")
    print("How well the damaged hemisphere still did its job (second half of the run, healthy = 100%):")
    print(f"  connected brain: {con['damaged_side_score']:.0%}")
    print(f"  split brain:     {spl['damaged_side_score']:.0%}")
    print(f"  -> help from the other hemisphere: {summary['benefit_of_callosum'] * 100:+.0f} percentage points")
    print(f"Saved results to {out_dir}")
    if not args.no_open:
        webbrowser.open(viewer.resolve().as_uri())


if __name__ == "__main__":
    main()
