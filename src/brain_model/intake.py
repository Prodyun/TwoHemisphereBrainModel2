"""PATIENT INTAKE (DETECT + IDENTIFY): turn a patient's data into a damage map on the 180 areas per side.

Three ways in, from best to simplest. Files are only READ; nothing is written into data/.

  from_mask(path)        a lesion mask (NIfTI) in MNI152 space, e.g. from the ATLAS dataset or from the
                         DeepISLES segmenter. Each area's damage = the share of its voxels inside the
                         lesion, using the HCP-MMP1 (Glasser) atlas in MNI space
                         (data/raw/area/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz). The same 180 labels
                         cover both hemispheres; x < 0 (MNI) is the left one. A mask on another grid is
                         sampled at the atlas voxels' positions (nearest voxel).
  from_table(rows)       a table of areas and values: `damage` (0..1) or `suvr` (tau-PET standardised
                         uptake ratio, Alzheimer's). SUVR -> damage = (SUVR - 1.2) / (max - 1.2), so
                         only areas above a positivity cut-off count (`cutoff`, default 1.2: an
                         ASSUMPTION; published cut-offs are roughly 1.2-1.4 and depend on the tracer and
                         region); the highest areas are the epicentres.
  from_symptoms(...)     no scan: a symptom checklist names the jobs that are hit; the side comes from the
                         symptom (e.g. weak right hand -> left hemisphere) or from an EEG Brain Symmetry
                         Index: BSI = |L - R| / (L + R) of slow-wave (delta) power, the side with more slow
                         waves being the damaged one (ELECTRA-STROKE, 2024: delta BSI AUC 0.91 for large
                         vessel occlusion). This is a rough first guess for the model, NOT a diagnosis.

Each returns a patient dict: {name, disorder, damage, source, notes}.
"""
import csv
import io
from pathlib import Path

import numpy as np

ATLAS = Path("data/raw/area/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz")
ATLAS_LABELS = Path("data/raw/area/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt")
MIN_SHARE = 0.02   # areas with less than 2% of their voxels inside the lesion are ignored


def _labels(path: Path = ATLAS_LABELS) -> dict[int, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            k, name = line.split(maxsplit=1)
            out[int(k)] = name.strip().removeprefix("L_").removeprefix("R_").removesuffix("_ROI")
    return out


def from_mask(path: str | Path, disorder: str = "stroke", atlas: Path = ATLAS, labels: Path = ATLAS_LABELS,
              threshold: float = 0.5) -> dict:
    import nibabel as nib
    at = nib.load(str(atlas))
    lab = np.rint(np.asarray(at.dataobj)).astype(int)
    mk = nib.load(str(path))
    mask = np.asarray(mk.dataobj, dtype=float)
    if mask.ndim == 4:
        mask = mask[..., 0]
    ijk = np.argwhere(lab > 0)
    world = at.affine @ np.c_[ijk, np.ones(len(ijk))].T                  # 4 x N, MNI millimetres
    if mask.shape == lab.shape and np.allclose(mk.affine, at.affine):
        inside = mask[tuple(ijk.T)] > threshold
    else:
        vox = np.rint(np.linalg.inv(mk.affine) @ world).astype(int)[:3]
        ok = np.all((vox >= 0) & (vox < np.array(mask.shape)[:, None]), axis=0)
        inside = np.zeros(len(ijk), bool)
        inside[ok] = mask[tuple(vox[:, ok])] > threshold
    names = _labels(labels)
    side = np.where(world[0] < 0, "L_", "R_")
    label = lab[tuple(ijk.T)]
    damage = {}
    for prefix in ("L_", "R_"):
        sel = side == prefix
        total = np.bincount(label[sel], minlength=max(names) + 1)
        hit = np.bincount(label[sel & inside], minlength=max(names) + 1)
        for k, nm in names.items():
            if total[k] and hit[k] / total[k] >= MIN_SHARE:
                damage[prefix + nm] = round(float(min(0.98, hit[k] / total[k])), 3)
    if not damage:
        raise ValueError("the mask does not overlap any cortical area (is it in MNI152 space?)")
    vol = float(inside.sum()) * abs(np.linalg.det(at.affine[:3, :3]))
    return {"name": f"Uploaded lesion ({Path(path).name})", "disorder": disorder, "damage": damage,
            "source": "lesion mask", "notes": [f"{len(damage)} areas touched; lesion inside the cortex about "
                                               f"{vol / 1000:.1f} ml"]}


def read_table(text: str) -> list[dict]:
    """CSV text with columns area, value (a header row is optional)."""
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if len(r) < 2 or not r[0].strip():
            continue
        try:
            rows.append({"area": r[0].strip(), "value": float(r[1])})
        except ValueError:
            continue   # header
    return rows


def from_table(rows: list[dict], kind: str = "damage", disorder: str = "stroke", cutoff: float = 1.2) -> dict:
    if kind not in ("damage", "suvr"):
        raise ValueError("kind must be 'damage' or 'suvr'")
    vals = {}
    for r in rows:
        a = str(r["area"]).removesuffix("_ROI")
        if not a.startswith(("L_", "R_")):
            raise ValueError(f"area {a} must start with L_ or R_")
        vals[a] = float(r["value"])
    if kind == "suvr":
        top = max(vals.values())
        if top <= cutoff:
            raise ValueError(f"no area is above the tau-PET cut-off (SUVR {cutoff})")
        damage = {a: round(min(0.98, (v - cutoff) / (top - cutoff) * 0.9), 3) for a, v in vals.items() if v > cutoff}
    else:
        if any(not 0 <= v <= 1 for v in vals.values()):
            raise ValueError("damage values must be between 0 and 1")
        damage = {a: round(min(0.98, v), 3) for a, v in vals.items() if v > 0}
    epic = sorted(damage, key=damage.get, reverse=True)[:3]
    return {"name": "Uploaded table", "disorder": disorder, "damage": damage, "source": f"table ({kind})",
            "notes": [f"highest: {', '.join(epic)}"]}


# symptom -> (side, job); "both" = memory in both hemispheres
SYMPTOMS = {
    "weak right hand or arm": ("left", "movement"),
    "weak left hand or arm": ("right", "movement"),
    "trouble speaking or understanding": ("left", "language"),
    "trouble with numbers or sequences": ("left", "math_sequences"),
    "ignores the left side of space": ("right", "spatial_attention"),
    "trouble recognising faces": ("right", "faces"),
    "flat tone of voice": ("right", "emotion_prosody"),
    "loss of right visual field": ("left", "vision"),
    "loss of left visual field": ("right", "vision"),
}


def symmetry_index(delta_left: float, delta_right: float) -> dict:
    """EEG Brain Symmetry Index of slow-wave power: 0 = symmetric, 1 = all on one side."""
    total = delta_left + delta_right
    if total <= 0:
        raise ValueError("EEG power must be positive")
    bsi = abs(delta_left - delta_right) / total
    side = "left" if delta_left > delta_right else "right"
    return {"bsi": round(bsi, 3), "side": side if bsi >= 0.1 else None}


def from_symptoms(symptoms: list[str], config: dict, severity: float = 0.7, eeg: dict | None = None,
                  disorder: str = "stroke") -> dict:
    """A first-guess damage map from symptoms (+ optional EEG asymmetry): the job's first areas at
    `severity`, the rest at half. Several symptoms on different sides use the EEG side if given."""
    picked = [SYMPTOMS[s] for s in symptoms if s in SYMPTOMS]
    if not picked and not eeg:
        raise ValueError(f"choose at least one symptom from {sorted(SYMPTOMS)} or give EEG power")
    notes = []
    eeg_side = None
    if eeg:
        si = symmetry_index(float(eeg["delta_left"]), float(eeg["delta_right"]))
        eeg_side = si["side"]
        notes.append(f"EEG symmetry index {si['bsi']:.2f}" + (f": more slow waves on the {eeg_side}" if eeg_side
                                                              else ": no clear asymmetry"))
    sides = {s for s, _ in picked}
    if len(sides) > 1 and eeg_side:
        picked = [(s, j) for s, j in picked if s == eeg_side]
        notes.append(f"symptoms point to both sides; using the EEG side ({eeg_side})")
    if not picked and eeg_side:
        picked = [(eeg_side, "movement")]
        notes.append("no symptom given: assuming the movement job on the EEG side")
    damage = {}
    for side, job in picked:
        funcs = config[f"{side}_hemisphere"]["functions"]
        if job not in funcs:
            continue
        areas = [str(a) for a in funcs[job]["areas"]]
        prefix = "L_" if side == "left" else "R_"
        for k, a in enumerate(areas):
            damage[prefix + a] = max(damage.get(prefix + a, 0.0), round(severity * (1.0 if k < 2 else 0.5), 3))
    if not damage:
        raise ValueError("no job found for these symptoms")
    notes.append("first guess from symptoms: confirm with a scan")
    return {"name": "Symptom check (no scan)", "disorder": disorder, "damage": damage, "source": "symptoms",
            "notes": notes}
