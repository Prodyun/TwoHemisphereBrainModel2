"""Simulate thousands of patients: the training data for the AI Help Predictor.

    python scripts/build_help_dataset.py --quick          # 40 areas, ~3 minutes (a check)
    python scripts/build_help_dataset.py                  # all 360 areas, ~30-60 minutes
    python scripts/build_help_dataset.py --seeds 2        # more noise seeds

Every one of the 360 areas is damaged (with its 4 most strongly wired neighbours), at 2 severities,
as a stroke and as a tumour, and treated with 5 practice modes; each case runs connected and split.
Finished cases are saved at once (results/help_dataset/rows.jsonl), so a stopped run continues
where it left off.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from brain_model.hcp import load_hcp_brain  # noqa: E402
from brain_model.help_data import MODES, case_key, run_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="40 spread-out areas, 1 seed")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--severities", type=float, nargs="+", default=[0.6, 0.9])
    ap.add_argument("--disorders", nargs="+", default=["stroke", "tumor"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="results/help_dataset")
    args = ap.parse_args()

    names = list(load_hcp_brain().names)
    if args.quick:
        names = names[::9]  # every 9th area: 40 areas on both sides
    cases = [(a, s, d, m, seed) for a in names for s in args.severities for d in args.disorders
             for m in MODES for seed in range(1, args.seeds + 1)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [c for c in cases if case_key(c) not in done]
    print(f"{len(cases)} cases ({len(cases) - len(todo)} already done) on {args.workers} cores", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool, open(path, "a", encoding="utf-8") as log:
        for i, row in enumerate(pool.imap_unordered(run_case, todo, chunksize=2), 1):
            log.write(json.dumps(row) + "\n")
            log.flush()
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  ({rate:.1f}/s, ~{(len(todo) - i) / rate / 60:.0f} min left)", flush=True)
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
