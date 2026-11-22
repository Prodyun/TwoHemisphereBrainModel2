"""Build the real-brain file the model uses:

    python scripts/fetch_enigma_data.py     # once: download the HCP group files
    python scripts/prepare_hcp.py           # writes data/processed/hcp_brain.npz
"""
import argparse

import numpy as np

from brain_model.hcp import load_hcp_brain
from brain_model.hcp.enigma import build_brain_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/enigma")
    parser.add_argument("--out", default="data/processed/hcp_brain.npz")
    args = parser.parse_args()

    path = build_brain_file(args.raw, args.out)
    brain = load_hcp_brain(path)
    lr = brain.sc_left_right
    print(f"Saved {path}")
    print(f"  {brain.n} areas per hemisphere")
    for side, sc in (("left", brain.sc_left), ("right", brain.sc_right)):
        print(f"  {side}: {len(brain.sides[side]['verts'])} surface points, "
              f"on average {np.count_nonzero(sc, axis=1).mean():.1f} connections per area")
    no_link = int((np.count_nonzero(lr, axis=1) == 0).sum())
    print(f"  callosum: {np.count_nonzero(lr)} left-right connections; "
          f"{no_link} left areas have no detected direct link to the right")


if __name__ == "__main__":
    main()
