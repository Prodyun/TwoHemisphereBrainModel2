# Two-Hemisphere Brain Model

An AI simulation of the brain as two cooperating hemispheres (left and right),
connected by a simulated corpus callosum.

## Project layout

```text
TwoHemisphereBrainModel2/
├── src/brain_model/          # The model itself (importable Python package)
│   ├── hemispheres/          #   base.py (shared), left.py, right.py
│   ├── connectivity/         #   corpus_callosum.py: link between hemispheres
│   ├── simulation/           #   runner.py: runs the brain over time
│   ├── utils/                #   config loading and helpers
│   └── brain.py              #   puts both hemispheres + callosum together
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
python scripts/run_simulation.py         # run a simulation with configs/default.yaml
```

## Running an experiment

1. Copy `configs/default.yaml` to a new file, e.g. `configs/split_brain.yaml`, and change settings
   (for example `corpus_callosum.strength: 0`).
2. Run `python scripts/run_simulation.py --config configs/split_brain.yaml`.
3. Results are saved to `results/<timestamp>/`.
4. Record what you tried and what you saw in `experiments/`.

See [docs/architecture.md](docs/architecture.md) for how the model works.
