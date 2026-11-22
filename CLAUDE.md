# Project instructions for Claude

Two-hemisphere (left/right) brain model, simulated with AI, in Python.

## Conventions
- Model code goes in `src/brain_model/`. Left- and right-specific behaviour goes in
  `hemispheres/left.py` and `hemispheres/right.py`; shared behaviour goes in `hemispheres/base.py`.
- Anything that passes information between hemispheres goes in `connectivity/`.
- Tunable numbers belong in `configs/*.yaml`, not hard-coded in Python.
- Command-line entry points go in `scripts/`. Outputs go in `results/` (git-ignored).
- Add or update tests in `tests/` with every change, and run `pytest` before finishing.
- Keep `README.md` and `docs/architecture.md` in sync when the structure changes.

## Requirements
- The owner's requirements live in `docs/requirements.md`. Read it before planning new work,
  and ask about any blank sections that matter for the task.

## User preferences
- The owner is new to this kind of project: explain choices in plain language.
- Add new instructions from the owner to this file so they carry across sessions.
- This is a science fair project aiming for ISEF level. Keep the science honest: state the model's
  assumptions and always compare against a control (the split brain).
- Visual style for the simulation: black background, and the left and right hemispheres shown as
  two separate 3D models that can each be rotated on their own.
- Make the two hemispheres as realistic as possible: genuinely different models (their own real
  wiring, settings and functions) that share some qualities, never mirror copies of each other.
- Interview the owner with questions before planning bigger work; don't assume.
- Working style: build the whole agreed chunk, then explain it in plain language at the end.

## Data files and the chat
- Never ask the owner to attach data files (CSV, .nii, .zip) to the chat: they cause "prompt is too
  long". Files stay on disk; the owner gives the path or says "downloaded", and Claude moves them with
  shell commands and inspects them only through small Python summaries (sizes, shapes, row counts).
  `.claude/settings.json` blocks the Read tool on `data/raw`, `data/processed` and scan/zip files.
- Long runs go through `scripts/run_pipeline.py`, started as a Windows scheduled task by
  `scripts/start_pipeline_task.ps1`, so they don't depend on Claude, VS Code or internet, and resume
  after a shutdown. Progress: `results/pipeline/status.txt`.

## Project facts (decided 2026-11-21)
- Timeline: finish within 2-3 days of 2026-11-21. Laptop, no GPU.
- Real data: HCP group data from the ENIGMA Toolbox (`scripts/fetch_enigma_data.py`), plus
  resting fMRI of 2 pilot subjects used only to check/fit the model.
- The GitHub repo is public: never commit HCP data, derived data, or subject IDs (refer to
  "subject A/B"). Work on branch `real-hcp-brain`, small commits, push periodically, PR into main.
