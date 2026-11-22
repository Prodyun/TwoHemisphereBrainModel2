"""Entry point:  python scripts/run_simulation.py --config configs/default.yaml"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from brain_model.simulation import run_simulation
from brain_model.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    history = run_simulation(config)

    out_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "history.npz", **history)

    corr = np.corrcoef(history["left"].ravel(), history["right"].ravel())[0, 1]
    print(f"Ran {config['steps']} steps. Left/right activity correlation: {corr:.3f}")
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
