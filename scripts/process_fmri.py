"""Turn each pilot subject's resting-state fMRI into a 360 x 360 functional connectivity matrix.

    python scripts/process_fmri.py

Reads every *.dtseries.nii in data/raw/hcp/<subject>/ for the subjects in
data/raw/pilot_subjects.txt and writes data/processed/subject_fc.npz with one matrix per subject,
named "A", "B", ... (no subject IDs, so results can be shared safely).
"""
import argparse
from pathlib import Path

import numpy as np

from brain_model.hcp import load_hcp_brain
from brain_model.hcp.fmri import subject_fc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="data/raw/pilot_subjects.txt")
    parser.add_argument("--raw", default="data/raw/hcp")
    parser.add_argument("--out", default="data/processed/subject_fc.npz")
    args = parser.parse_args()

    brain = load_hcp_brain()
    regions = {s: brain.sides[s]["vertex_region"].astype(int) for s in ("left", "right")}
    ids = Path(args.subjects).read_text(encoding="utf-8").split()
    out = {}
    for label, sid in zip("ABCDEFGHIJ", ids):
        runs = sorted((Path(args.raw) / sid).rglob("*.dtseries.nii"))
        if not runs:
            print(f"subject {label}: no .dtseries.nii files in {Path(args.raw) / sid}, skipped")
            continue
        print(f"subject {label}: {len(runs)} run(s) ...")
        out[label] = subject_fc(runs, regions, brain.n)
        group = brain.fc
        iu = np.triu_indices(2 * brain.n, 1)
        r = np.corrcoef(out[label][iu], group[iu])[0, 1]
        print(f"  similarity to the HCP group FC: r = {r:.2f}")
    if not out:
        raise SystemExit("No fMRI found. See scripts/select_subjects.py for what to download.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
