"""Run the whole real-data pipeline from start to finish, unattended, and pick up where it stopped.

    python scripts/run_pipeline.py            # usually started by Windows Task Scheduler (see below)
    python scripts/run_pipeline.py --redo     # start again from scratch

Steps (each is skipped once it has finished, so a shutdown or crash loses no finished step):
  1. process_fmri   the pilot subjects' resting fMRI -> data/processed/subject_fc.npz
  2. fit            fit left/right/callosum settings to real fMRI, write them into configs/hcp.yaml
  3. batch          all experiments; every finished run is saved at once, so it resumes mid-way
  4. figures        poster figures and summary.md

Progress: results/pipeline/status.txt (one line) and results/pipeline/log.txt (everything).
Needs no internet. While it runs, it asks Windows not to go to sleep on its own.

To run it outside VS Code / Claude, and restart it automatically after a reboot:
  powershell -ExecutionPolicy Bypass -File scripts/start_pipeline_task.ps1
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "results" / "pipeline"
FIT_DIR = ROOT / "results" / "fit_main"
BATCH_DIR = ROOT / "results" / "batch_main"
FIG_DIR = ROOT / "results" / "figures_main"

STEPS = [  # (name, command arguments, the file that proves the step is finished)
    ("process_fmri", ["scripts/process_fmri.py"], ROOT / "data" / "processed" / "subject_fc.npz"),
    ("fit", ["scripts/fit_to_fmri.py", "--apply", "--out", str(FIT_DIR)], FIT_DIR / "fit.json"),
    ("batch", ["scripts/run_batch.py", "--out", str(BATCH_DIR)], BATCH_DIR / "results.csv"),
    ("figures", ["scripts/make_figures.py", "--batch", str(BATCH_DIR), "--fit", str(FIT_DIR),
                 "--out", str(FIG_DIR)], FIG_DIR / "summary.md"),
]


def stay_awake(on: bool) -> None:
    """Ask Windows not to sleep while we work (the screen may still turn off). No-op elsewhere."""
    if os.name != "nt":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def say(msg: str, log) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def status(msg: str) -> None:
    (PIPE / "status.txt").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true", help="delete earlier pipeline outputs first")
    args = parser.parse_args()
    PIPE.mkdir(parents=True, exist_ok=True)
    done_flag = PIPE / "DONE"
    if args.redo:
        for p in (FIT_DIR, BATCH_DIR, FIG_DIR):
            shutil.rmtree(p, ignore_errors=True)
        done_flag.unlink(missing_ok=True)
        for m in PIPE.glob("step_*.done"):
            m.unlink()
        (ROOT / "data" / "processed" / "subject_fc.npz").unlink(missing_ok=True)
    if done_flag.exists():
        print("Pipeline already finished (results/pipeline/DONE). Use --redo to run again.")
        return

    lock = PIPE / "running.lock"  # stops two copies running at once (e.g. logon + manual start)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pid = lock.read_text().strip()
        if pid.isdigit() and _alive(int(pid)):
            print(f"Already running (process {pid}).")
            return
        lock.write_text(str(os.getpid()))  # left over from a shutdown: take it over

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src"))
    stay_awake(True)
    try:
        with open(PIPE / "log.txt", "a", encoding="utf-8") as log:
            say("pipeline started", log)
            for i, (name, cmd, output) in enumerate(STEPS, 1):
                marker = PIPE / f"step_{name}.done"
                if marker.exists() and output.exists():
                    say(f"step {i}/{len(STEPS)} {name}: already done, skipped", log)
                    continue
                status(f"running step {i}/{len(STEPS)}: {name}")
                say(f"step {i}/{len(STEPS)} {name}: started", log)
                t0 = time.time()
                proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT)
                minutes = (time.time() - t0) / 60
                if proc.returncode != 0 or not output.exists():
                    status(f"FAILED at step {i}/{len(STEPS)}: {name} (see results/pipeline/log.txt)")
                    say(f"step {name} FAILED (exit {proc.returncode}) after {minutes:.1f} min", log)
                    sys.exit(1)
                marker.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
                say(f"step {i}/{len(STEPS)} {name}: finished in {minutes:.1f} min", log)
            done_flag.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
            status(f"DONE. Figures and summary in {FIG_DIR.relative_to(ROOT).as_posix()}/")
            say("pipeline finished", log)
    finally:
        stay_awake(False)
        lock.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    try:
        main()
    except Exception:  # with no console window, a crash would be invisible: write it down
        import traceback
        PIPE.mkdir(parents=True, exist_ok=True)
        with open(PIPE / "log.txt", "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        status("FAILED: crashed (see results/pipeline/log.txt)")
        raise
