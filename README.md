# Two-Hemisphere Brain Model

A simulation of the brain as two distinct hemispheres (left and right), linked by a
simulated corpus callosum. A disorder starts in one hemisphere and spreads. The project tests
whether signals from the healthy hemisphere help the damaged one keep working, comparing a
connected brain with a split brain. Results appear in a 3D viewer: two separate hemispheres
on a black screen that you can rotate independently.

## Project layout

```text
TwoHemisphereBrainModel2/
├── src/brain_model/          # The model itself (importable Python package)
│   ├── anatomy.py            #   hemisphere shape, 180 regions, wiring between them
│   ├── hemispheres/          #   base.py (shared), left.py, right.py
│   ├── connectivity/         #   corpus_callosum.py: link between hemispheres + plasticity
│   ├── lesion.py             #   the disorder that starts in one region and spreads
│   ├── simulation/           #   runner.py: runs the brain, the experiment and the scoring
│   ├── visualization/        #   the 3D viewer (black screen, two rotatable hemispheres)
│   ├── utils/                #   config loading and helpers
│   └── brain.py              #   puts hemispheres + callosum + lesion together
├── configs/                  # YAML settings for each experiment (default.yaml)
├── scripts/                  # Command-line entry points (run_simulation.py)
├── experiments/              # Write-ups of individual experiments
├── notebooks/                # Jupyter notebooks for exploration and plots
├── tests/                    # Automated tests (pytest)
├── data/raw/                 # Original input data (not committed to git)
├── data/processed/           # Cleaned/derived data (not committed to git)
├── results/                  # Simulation outputs, one folder per run (not committed)
└── docs/                     # Design notes (architecture.md)
```

## Getting started (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

pytest                                   # run the tests
python scripts/run_simulation.py         # run the experiment and open the 3D viewer
```

The script prints how well the damaged hemisphere still works with and without the corpus
callosum. It then opens `results/<timestamp>/viewer.html` in your browser (an internet
connection is needed the first time, to load the 3D library). In the viewer:

- drag a hemisphere to rotate it, and scroll to zoom; each one moves on its own
- **Connected brain / Split brain** switches between the test and the control
- colors: blue = activity, red = damage, gold = signals arriving from the healthy side
- add `#frame=150&paused` to the end of the address to open at a specific moment (useful for screenshots)

## Running an experiment

1. Copy `configs/default.yaml` to a new file, e.g. `configs/split_brain.yaml`, and change settings
   (for example `lesion.hemisphere: right`, or `lesion.cross_spread: 0.5`).
2. Run `python scripts/run_simulation.py --config configs/split_brain.yaml`.
3. Results are saved to `results/<timestamp>/`.
4. Record what you tried and what you saw in `experiments/`.

See [docs/architecture.md](docs/architecture.md) for how the model works.
