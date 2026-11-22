# Data sources

No brain data is stored in this repository (it is public, and HCP's data-use terms do not allow
redistributing the data). Everything below is downloaded to `data/raw/` and processed into
`data/processed/`, both git-ignored.

## 1. Real brain shape, areas and wiring: ENIGMA Toolbox (no login)

```powershell
python scripts/fetch_enigma_data.py   # downloads ~4 MB, pinned to one ENIGMA version
python scripts/prepare_hcp.py         # builds data/processed/hcp_brain.npz
```

| File | What it is | Used for |
|---|---|---|
| `conte69_32k_lh/rh.gii` | Group-average cortical surfaces (32k vertices per side, MNI space) | 3D viewer shape; area centres |
| `glasser_360_conte69.csv` | Glasser HCP-MMP1.0 atlas on that surface (180 areas per side) | which area each surface point belongs to |
| `strucMatrix_ctx_glasser_360.csv` | HCP group **structural** connectome from diffusion MRI (log-scaled fibre counts) | the model's wiring: left-left, right-right, left-right |
| `funcMatrix_ctx_glasser_360.csv` | HCP group **functional** connectivity from resting fMRI | a fitting target |

Facts worth knowing (printed by `prepare_hcp.py`):
- Each area has about 20 connections inside its own hemisphere.
- Only 827 left-right connections were detected. **78 of the 180 left areas, including Broca's
  area (L_44), have no detected direct link to the right hemisphere.** Diffusion tractography is
  known to miss many callosal fibres that reach the lateral cortex, so the "real" callosum
  underestimates the true connections. The experiments therefore also run a mirror-only and a
  real-plus-mirror callosum.
- 63 vertices on the right surface carry a stray left-hemisphere label; they are treated as
  having no area.

Citations:
- Larivière S. et al. (2021) The ENIGMA Toolbox: multiscale neural contextualization of
  multisite neuroimaging datasets. *Nature Methods* 18:698-700.
- Glasser M.F. et al. (2016) A multi-modal parcellation of human cerebral cortex. *Nature* 536:171-178.
- Van Essen D.C. et al. (2013) The WU-Minn Human Connectome Project: an overview. *NeuroImage* 80:62-79.

## 2. Pilot subjects' resting-state fMRI (ConnectomeDB login)

```powershell
python scripts/select_subjects.py --csv data/raw/HCP_YA_subjects.csv --n 2
```
This picks 2 subjects with complete 3T scans and no quality-control issue (fixed seed) and writes
their IDs to `data/raw/pilot_subjects.txt`. In code, docs and results they are only called
"subject A" and "subject B".

For each subject, download from ConnectomeDB (HCP Young Adult, package "Resting State fMRI
FIX-Denoised (Compact)"):
- `rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii`
- `rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii`

Put them in `data/raw/hcp/<subject id>/`, then:
```powershell
python scripts/process_fmri.py    # -> data/processed/subject_fc.npz
python scripts/fit_to_fmri.py     # fits the model to the group and to each subject
```
