# Two-Hemisphere Brain Model

A simulation of the brain as two distinct hemispheres (left and right), built on real Human
Connectome Project (HCP) data and linked by a simulated corpus callosum. A disorder starts in one
hemisphere and spreads. The project tests whether signals from the healthy hemisphere help the
damaged one keep working, comparing a connected brain with a split brain (the control).
Results appear in a 3D viewer: two separate hemispheres on a black screen that you can rotate
independently.

The two hemispheres are different models. Each has its own real wiring, its own settings and its
own jobs: language and math/sequences on the left; spatial attention, faces and emotion in the voice
on the right. Both also do vision and movement, each for the opposite side of the body. See
[docs/architecture.md](docs/architecture.md).

## Project layout

```text
TwoHemisphereBrainModel2/
├── src/brain_model/          # The model itself (importable Python package)
│   ├── anatomy.py            #   real-vs-stand-in brain, wiring helpers
│   ├── hcp/                  #   real HCP data: download/prepare (enigma.py), load, fMRI
│   ├── hemispheres/          #   base.py (shared + job patterns), left.py, right.py
│   ├── connectivity/         #   corpus_callosum.py: link between hemispheres + plasticity
│   ├── lesion.py             #   the disorder: spreading, stroke, tumour, or a patient's damage map
│   ├── recovery.py           #   healing driven by use (practice), with the golden-period window
│   ├── therapy.py            #   practice (physical, imagery, BCI, sleep replay, VNS) + stimulation
│   ├── patient.py            #   a patient file -> model config; the IDENTIFY problem card
│   ├── planning.py           #   RECRUIT / DOSE / TIME searches for a personal plan
│   ├── simulation/           #   runner.py (experiment), outcome.py (patient measures), rest.py + fit.py
│   ├── visualization/        #   the 3D viewer (black screen, two rotatable hemispheres)
│   ├── utils/                #   config loading
│   └── brain.py              #   puts hemispheres + callosum + lesion together
├── configs/                  # hcp.yaml (real brain), default.yaml (stand-in), care.yaml (patient care),
│                             #   patients/*.yaml (example patients)
├── scripts/                  # command-line entry points (see below)
├── experiments/              # write-ups of individual experiments
├── tests/                    # automated tests (pytest)
├── data/raw/, data/processed/  # downloaded and derived data (not committed to git)
├── results/                  # simulation outputs, one folder per run (not committed)
└── docs/                     # architecture.md, data.md, requirements.md, factors.md
```

## Getting started (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/fetch_enigma_data.py      # download the real brain (HCP group data, ~4 MB, no login)
python scripts/prepare_hcp.py            # build data/processed/hcp_brain.npz
pytest                                   # run the tests
python scripts/run_simulation.py --config configs/hcp.yaml   # run the experiment, open the 3D viewer
```

The script prints how well the damaged hemisphere still works with and without the corpus
callosum, overall and for each job. It then opens `results/<timestamp>/viewer.html` in your browser
(an internet connection is needed the first time, to load the 3D library). In the viewer:

- **Separate / Together** (top): two hemispheres you rotate on their own, or both in one scene
  with the 150 strongest real callosal fibres drawn between them. Pulses run along each fibre in
  the direction the signal travels: gold = into the damaged side, blue = the other way
- drag to rotate, scroll to zoom; point at an area to see its name, its job, its damage and the
  input it gets from the other side
- **Connected brain / Split brain** switches between the test and the control
- **Colour by**: *Damage* (grey = healthy, yellow -> red -> dark red; an area flashes when the
  disease first reaches it, and the connections carrying it glow orange), *Input from other side*
  (gold = signals arriving, blue = signals being sent), or *Activity*
- **Area outlines** shows the 180 areas the model is really made of; **Network view** shows them as
  nodes with their strongest real connections; **Show functions** colours each job's areas
- add `#frame=150&paused&view=together&color=input&network` to the end of the address to open at a
  specific moment and setting (useful for screenshots)

## Choose where the disorder starts (any of the 360 areas)

```powershell
python scripts/explore.py        # or double-click explore.bat
```

This opens the viewer through a small helper program that runs on your laptop only. Click any area
on either hemisphere (left or right), choose **Spreading** or **Stroke**, and press **Start disorder
here**. About 3 seconds later the new run plays: the chosen hemisphere is damaged, and the other one
helps through the corpus callosum (compare with **Split brain**). Keep the helper's window open while
you use the viewer; press Ctrl+C there to stop. Adding `#pick=right:R_FFC&run` to the address picks
and starts an area without clicking.

`python scripts/support_map.py` runs every one of the 360 start areas and maps how much the healthy
hemisphere helps for each (`results/support_map/`); it is also step 5 of the full pipeline.

## Full pipeline (for the results and poster)

```powershell
python scripts/select_subjects.py --csv data/raw/HCP_YA_subjects.csv   # pick 2 pilot subjects
# download their resting fMRI (see docs/data.md), then:
python scripts/process_fmri.py      # real fMRI -> connectivity per subject
python scripts/fit_to_fmri.py       # fit left/right settings to real fMRI, with controls (~15 min)
python scripts/run_batch.py         # all experiments x 20 seeds x 20 start areas (~10 min)
python scripts/make_figures.py      # poster figures + results/figures_*/summary.md
```

## A personal plan for a patient ("Brain SWAT")

A patient file lists the damaged areas (any of the 360 Glasser areas, either side) and the disorder:

```yaml
# configs/patients/left_6d_stroke.yaml
name: "Left 6d stroke (example)"
disorder: stroke            # stroke | tumor | spreading
damage: {L_6d: 0.95, L_6a: 0.7, L_4: 0.8, L_3a: 0.7}
```

```powershell
python scripts/therapy_plan.py --patient configs/patients/left_6d_stroke.yaml   # ~2 minutes
python scripts/find_targets.py --patient configs/patients/left_6d_stroke.yaml   # only "where to help"
```

The plan goes step by step:
- **Identify** what is damaged and which jobs are hit.
- **Recruit:** which healthy-side areas to stimulate.
- **Dose:** which practice to use (physical, imagery, BCI, sleep replay, VNS) and how strong.
- **Time:** the golden period.
- **Predict** the recovery against standard care, today's standard target, no therapy and the split brain.

Three measures are reported: physical repetitions to the goal, weeks to the goal, and final
function. Output goes to `results/plans/<patient>/personal_plan.md` with two figures. Every
adjustable factor is listed in [docs/factors.md](docs/factors.md).

## Running your own experiment

1. Copy `configs/hcp.yaml` to a new file, e.g. `configs/stroke_right.yaml`, and change settings
   (for example `lesion.hemisphere: right`, `lesion.region: R_FFC`, `lesion.type: stroke`, or
   `corpus_callosum.wiring: mirror`).
2. Run `python scripts/run_simulation.py --config configs/stroke_right.yaml`.
3. Results are saved to `results/<timestamp>/`.
4. Record what you tried and what you saw in `experiments/`.

Data sources and citations: [docs/data.md](docs/data.md).
