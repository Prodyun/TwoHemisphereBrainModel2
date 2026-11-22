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
