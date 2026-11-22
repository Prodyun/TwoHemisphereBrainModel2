"""Pick the pilot subjects from the HCP-YA subject table and say exactly which files to download.

    python scripts/select_subjects.py --csv data/raw/HCP_YA_subjects.csv --n 2

Keeps subjects with complete 3T scans (structural, resting fMRI, diffusion) and no known quality
issue, then picks at random (fixed seed, so the choice is reproducible). Prefers one female and one
male. Subject IDs are written only to data/raw/pilot_subjects.txt, which git ignores (the repo is
public, and HCP data-use terms ask us not to share subject data).
"""
import argparse
import csv
import random
from pathlib import Path

RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL"]


def eligible(row: dict) -> bool:
    return (row.get("3T_Full_MR_Compl") == "true"
            and row.get("3T_RS-fMRI_PctCompl") == "100.0"
            and row.get("3T_dMRI_Compl") == "true"
            and not row.get("QC_Issue", "").strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/HCP_YA_subjects.csv")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw/pilot_subjects.txt")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if eligible(r)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    chosen, by_sex = [], {}
    for r in rows:  # alternate sexes where possible
        by_sex.setdefault(r["Gender"], []).append(r)
    pools = [p for p in by_sex.values() if p]
    while len(chosen) < args.n and any(pools):
        for p in pools:
            if p and len(chosen) < args.n:
                chosen.append(p.pop(0))

    Path(args.out).write_text("\n".join(r["Subject"] for r in chosen) + "\n", encoding="utf-8")
    print(f"{len(rows)} subjects qualify. Chosen ({args.out}):")
    for label, r in zip("ABCDEFGHIJ", chosen):
        print(f"  subject {label}: {r['Subject']}  ({r['Gender']}, age {r['Age']})")
    print("\nDownload for each subject (ConnectomeDB > HCP Young Adult > package"
          " 'Resting State fMRI FIX-Denoised (Compact)'), then put the files in"
          " data/raw/hcp/<subject>/:")
    for run in RUNS:
        print(f"  {run}_Atlas_MSMAll_hp2000_clean.dtseries.nii")


if __name__ == "__main__":
    main()
