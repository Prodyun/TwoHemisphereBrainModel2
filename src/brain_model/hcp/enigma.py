"""Turn the ENIGMA Toolbox's HCP files into one compact brain file the model can load.

Input (from scripts/fetch_enigma_data.py, in data/raw/enigma/):
  * conte69_32k_lh/rh.gii                 real cortical surfaces (millimetres, MNI space)
  * glasser_360_conte69.csv               Glasser area of every surface vertex (0 = medial wall)
  * strucMatrix_ctx_glasser_360.csv       HCP group structural connectome (diffusion MRI), 360 x 360
  * strucLabels_ctx_glasser_360.csv       area names, in matrix order (L_V1 ... L_p24, R_V1 ... R_p24)
  * funcMatrix_ctx_glasser_360.csv        HCP group resting-state functional connectivity, 360 x 360

Output: data/processed/hcp_brain.npz with, for each hemisphere, the mesh, the area of every vertex,
area centres, and the wiring blocks left-left, right-right and left-right.

Units: the model uses the stand-in's units (1 unit = 100 mm), with x = 0 at the midline.
"""
from pathlib import Path

import numpy as np

N_PER_SIDE = 180
MM_PER_UNIT = 100.0


def read_labels(path: str | Path) -> list[str]:
    return [s.strip() for s in Path(path).read_text(encoding="utf-8").replace("\n", ",").split(",") if s.strip()]


def read_matrix(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def read_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib  # only needed when preparing data
    gii = nib.load(str(path))
    verts = gii.darrays[0].data.astype(float)
    faces = gii.darrays[1].data.astype(np.int64)
    return verts, faces


def connection_strength(raw: np.ndarray) -> np.ndarray:
    """ENIGMA's weights are log-scaled fibre counts (0 = no connection; a few are slightly negative).

    Map every existing connection to a positive strength between 0.05 (weakest) and 1 (strongest),
    keeping their order. Missing connections stay 0, and nothing connects to itself."""
    w = np.zeros_like(raw, dtype=float)
    present = raw != 0
    if present.any():
        v = raw[present]
        span = v.max() - v.min()
        w[present] = 0.05 + 0.95 * ((v - v.min()) / span if span > 0 else 1.0)
    np.fill_diagonal(w, 0.0)
    return w


def vertex_regions(labels: np.ndarray, first: int, n: int) -> np.ndarray:
    """Atlas numbers -> area index 0..n-1 for one hemisphere; -1 = no area (medial wall).

    ENIGMA numbers left areas 1..180 and right areas 181..360. A few right-surface vertices
    carry a stray left number; they are treated as having no area."""
    idx = labels - first
    return np.where((idx >= 0) & (idx < n), idx, -1)


def region_geometry(verts: np.ndarray, vertex_region: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre (mean position) of each area, and an 'anchor': the area's surface vertex nearest its centre."""
    centers = np.zeros((n, 3))
    anchors = np.zeros((n, 3))
    for r in range(n):
        pts = verts[vertex_region == r]
        if len(pts) == 0:
            raise ValueError(f"area {r} has no vertices on the surface")
        centers[r] = pts.mean(axis=0)
        anchors[r] = pts[np.argmin(np.linalg.norm(pts - centers[r], axis=1))]
    return centers, anchors


def build_brain_file(raw_dir: str | Path, out_path: str | Path) -> Path:
    raw = Path(raw_dir)
    names = read_labels(raw / "strucLabels_ctx_glasser_360.csv")
    n = len(names) // 2
    if len(names) != 2 * n or not all(s.startswith("L_") for s in names[:n]) \
            or not all(s.startswith("R_") for s in names[n:]):
        raise ValueError("expected left areas (L_...) followed by right areas (R_...)")
    if [s[2:] for s in names[:n]] != [s[2:] for s in names[n:]]:
        raise ValueError("left and right areas must be in the same order (area i <-> area i)")

    sc = connection_strength(read_matrix(raw / "strucMatrix_ctx_glasser_360.csv"))
    fc = read_matrix(raw / "funcMatrix_ctx_glasser_360.csv")
    parc = np.loadtxt(raw / "glasser_360_conte69.csv").astype(int)

    sides = {}
    for key, fname in (("left", "conte69_32k_lh.gii"), ("right", "conte69_32k_rh.gii")):
        verts, faces = read_surface(raw / fname)
        sides[key] = {"verts": verts, "faces": faces}
    nv = len(sides["left"]["verts"])
    sides["left"]["region"] = vertex_regions(parc[:nv], first=1, n=n)
    sides["right"]["region"] = vertex_regions(parc[nv:], first=n + 1, n=n)

    # Shared centring so the two hemispheres keep their true positions relative to each other.
    all_verts = np.vstack([sides["left"]["verts"], sides["right"]["verts"]])
    shift = np.array([0.0, *all_verts[:, 1:].mean(axis=0)])

    out = {"names": np.array(names), "fc": fc,
           "sc_left": sc[:n, :n], "sc_right": sc[n:, n:], "sc_left_right": sc[:n, n:]}
    for key, s in sides.items():
        verts = (s["verts"] - shift) / MM_PER_UNIT
        centers, anchors = region_geometry(verts, s["region"], n)
        out.update({f"{key}_verts": verts.astype(np.float32), f"{key}_faces": s["faces"].astype(np.int32),
                    f"{key}_vertex_region": s["region"].astype(np.int16),
                    f"{key}_centers": centers, f"{key}_anchors": anchors})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out_path
