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

For each subject, download the **HCP-Young Adult 2025** resting-state package from ConnectomeDB on
BALSA (balsa.wustl.edu > ConnectomeDB tab; the old db.humanconnectome.org closed on 2025-10-01).
The package used here is `<subject>_Rest3TRecommended.zip` (about 10.8 GB per subject). Only its 4
per-run surface files are needed (418 MB each):
- `rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii` (also `REST1_RL`, `REST2_LR`, `REST2_RL`)

Do **not** also extract the concatenated `rfMRI_REST_Atlas_..._dtseries.nii`: it holds the same 4 runs
joined together, and `process_fmri.py` reads every `.dtseries.nii` in the folder, so it would be counted twice.
Never attach these files (or the subject CSV) to a Claude chat: they are far too large for a prompt.
Leave them on disk and give the path.

Put them in `data/raw/hcp/<subject id>/`, then:
```powershell
python scripts/process_fmri.py    # -> data/processed/subject_fc.npz
python scripts/fit_to_fmri.py     # fits the model to the group and to each subject
```

## 3. Real properties of each area, for the AI Help Predictor (no login)

```powershell
python scripts/fetch_area_data.py      # ~122 MB into data/raw/area/ (pinned versions)
python scripts/prepare_area_atlas.py   # -> data/processed/area_atlas.npz (one row per area)
```

| Data | Source | Licence / use |
|---|---|---|
| Gene activity per Glasser area: plasticity genes BDNF, GAP43, NTRK2, CAMK2A, GRIN2B, SYP; vulnerability genes MAPT, SNCA, TARDBP, APOE | Allen Human Brain Atlas (Hawrylycz 2012), Glasser-360 version from enigma-extra (Larivière 2021) | open; cite both |
| Network of each area (12 networks) | Cole-Anticevic Brain-wide Network Partition (Ji et al. 2019, *NeuroImage*) | non-commercial licence |
| Glasser 2016's 22 cortical sections, area volume | HCP-MMP1 region list (D. Patterson, Univ. Arizona) | no licence stated: cite, don't redistribute |
| Volumetric Glasser atlas (MNI) | Horn 2016, figshare 3501911 | CC BY 4.0 |
| Receptor/transporter PET densities: VAChT, NET, D1, D2, DAT, GABA-A, NMDA | Hansen et al. 2022, *Nat Neurosci* (netneurolab) | cite; check each map's licence |

Notes:
- 63 areas have a missing gene value (the Allen atlas did not sample every area), and 6 have a missing receptor value. The AI model handles missing values.
- The PET maps were made on slightly different MNI templates than the atlas. At 1–3 mm this mismatch is small compared with an area.
