"""Build ONE table with what is known about each of the 360 Glasser areas (the AI model's real data).

    python scripts/fetch_area_data.py      # once
    python scripts/prepare_area_atlas.py   # -> data/processed/area_atlas.npz

Columns (360 rows, same order as the model's area names L_... then R_...):
  network   which of 12 brain networks the area belongs to (Cole-Anticevic partition, Ji 2019)
  section   which of Glasser 2016's 22 cortical sections it is in (e.g. Primary_Visual), and its volume
  genes     Allen Human Brain Atlas expression (z-scored across areas) of plasticity genes
            (BDNF, GAP43, NTRK2, CAMK2A, GRIN2B, SYP) and disease-vulnerability genes (MAPT, SNCA,
            TARDBP, APOE)
  receptors PET density of 7 receptor/transporter maps (Hansen 2022), averaged over each area's
            voxels in the volumetric Glasser atlas (Horn 2016), z-scored across areas
The volumetric atlas uses the same 180 labels for both hemispheres, so voxels are split into left /
right by the sign of their MNI x coordinate. The PET maps were made on slightly different MNI
templates than the atlas; at 1-3 mm that mismatch is small compared with an area (~10-20 mm).
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brain_model.hcp import load_hcp_brain  # noqa: E402

RAW = Path("data/raw/area")
OUT = Path("data/processed/area_atlas.npz")
GENES = ["BDNF", "GAP43", "NTRK2", "CAMK2A", "GRIN2B", "SYP", "MAPT", "SNCA", "TARDBP", "APOE"]
RECEPTORS = {"VAChT": "VAChT_feobv_hc18_aghourian_sum.nii", "NET": "NAT_MRB_hc77_ding.nii.gz",
             "D1": "D1_SCH23390_hc13_kaller.nii", "D2": "D2_fallypride_hc49_jaworska.nii",
             "DAT": "DAT_fpcit_hc174_dukart_spect.nii", "GABAa": "GABAa-bz_flumazenil_hc16_norgaard.nii",
             "NMDA": "NMDA_ge179_hc29_galovic.nii.gz"}


def zscore(v: np.ndarray) -> np.ndarray:
    m, s = np.nanmean(v), np.nanstd(v)
    return (v - m) / s if s > 0 else v * 0


def networks(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lines = (RAW / "CABNP_LabelKey.txt").read_text().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]
    by_name = {r["GLASSERLABELNAME"].removesuffix("_ROI"): r["NETWORK"] for r in rows if r["GLASSERLABELNAME"]}
    net = np.array([by_name.get(n, "") for n in names])
    return net, np.array(sorted(set(net) - {""}))


def sections(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    text = (RAW / "HCP-MMP1_UniqueRegionList.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    by_name = {f"{r['LR']}_{r['region']}": r for r in rows}
    sec = np.array([by_name[n]["cortex"] if n in by_name else "" for n in names])
    vol = np.array([float(by_name[n]["volmm"]) if n in by_name else np.nan for n in names])
    return sec, vol


def genes(names: list[str]) -> np.ndarray:
    z = zipfile.ZipFile(RAW / "allgenes_stable_r0.2_glasser_360.csv.zip")
    with z.open(z.namelist()[0]) as f:
        rows = list(csv.reader(io.TextIOWrapper(f)))
    head = rows[0]
    cols = [head.index(g) for g in GENES]
    by_name = {r[0]: r for r in rows[1:]}
    out = np.full((len(names), len(GENES)), np.nan)
    for i, n in enumerate(names):
        if n in by_name:
            out[i] = [float(by_name[n][c]) if by_name[n][c] not in ("", "nan") else np.nan for c in cols]
    return np.column_stack([zscore(out[:, j]) for j in range(out.shape[1])])


def receptors(names: list[str]) -> np.ndarray:
    atlas = nib.load(RAW / "HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz")
    lab = np.asarray(atlas.dataobj).astype(int)
    labels = [ln.split(" ", 1)[1].strip().removesuffix("_ROI")[2:]
              for ln in (RAW / "HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt").read_text().splitlines() if ln.strip()]
    ijk = np.argwhere(lab > 0)
    xyz = nib.affines.apply_affine(atlas.affine, ijk)
    lab_v = lab[tuple(ijk.T)]
    side = np.where(xyz[:, 0] < 0, "L", "R")
    area = np.array([f"{s}_{labels[v - 1]}" for s, v in zip(side, lab_v)])
    index = {n: i for i, n in enumerate(names)}
    rows = np.array([index.get(a, -1) for a in area])
    out = np.full((len(names), len(RECEPTORS)), np.nan)
    for j, fname in enumerate(RECEPTORS.values()):
        img = nib.load(RAW / fname)
        data = np.asarray(img.dataobj, dtype=float).squeeze()
        vox = np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)).astype(int)
        inside = np.all((vox >= 0) & (vox < data.shape[:3]), axis=1) & (rows >= 0)
        vals = np.full(len(xyz), np.nan)
        vals[inside] = data[tuple(vox[inside].T)]
        vals[vals <= 0] = np.nan  # outside the PET brain mask
        sums = np.bincount(rows[inside & ~np.isnan(vals)], weights=vals[inside & ~np.isnan(vals)], minlength=len(names))
        counts = np.bincount(rows[inside & ~np.isnan(vals)], minlength=len(names))
        out[:, j] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    unmatched = sorted(set(area[rows < 0]))
    if unmatched:
        print(f"  note: {len(unmatched)} atlas labels not in the model's names, e.g. {unmatched[:5]}")
    return np.column_stack([zscore(out[:, j]) for j in range(out.shape[1])])


def main() -> None:
    brain = load_hcp_brain()
    names = list(brain.names)
    net, net_names = networks(names)
    sec, vol = sections(names)
    g = genes(names)
    r = receptors(names)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, names=np.array(names), network=net, network_names=net_names, section=sec,
                        volume=vol, genes=g, gene_names=np.array(GENES), receptors=r,
                        receptor_names=np.array(list(RECEPTORS)))
    print(f"Saved {OUT}: {len(names)} areas")
    print(f"  network: {np.sum(net != '')}/360 matched; {len(net_names)} networks: {', '.join(net_names)}")
    print(f"  section: {np.sum(sec != '')}/360 matched; {len(set(sec) - {''})} sections")
    print(f"  genes:   {int(np.isnan(g).any(axis=1).sum())} areas with a missing gene value")
    print(f"  receptors: {int(np.isnan(r).any(axis=1).sum())} areas with a missing receptor value")
    i = names.index("L_4")
    print(f"  e.g. L_4: network {net[i]}, section {sec[i]}, BDNF z={g[i, 0]:+.2f}, NMDA z={r[i, -1]:+.2f}")


if __name__ == "__main__":
    main()
