"""Builds the interactive 3D viewer: one HTML file showing both hemispheres on a black screen.

Each hemisphere is its own 3D model with its own camera, so you can rotate them separately.
The simulation data is embedded in the HTML, so the file opens directly in a browser
(it needs an internet connection the first time to load the Three.js 3D library).
"""
import json
from pathlib import Path

import numpy as np

from .. import anatomy
from ..simulation.runner import rolling_function_score

TEMPLATE = Path(__file__).with_name("viewer_template.html")
SIDES = {"left": -1, "right": 1}


def _to_three(p: np.ndarray) -> np.ndarray:
    """Model coords (x side, y front, z up) -> Three.js coords (x, y up, z toward viewer)."""
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def _ints(a: np.ndarray, scale: float = 100) -> list:
    return np.round(np.asarray(a) * scale).astype(int).tolist()


def _mesh_payload(config: dict, side_name: str) -> dict:
    """The 3D surface of one hemisphere and which brain region colours each surface point.

    Stand-in: a smooth hand-made surface; each point blends its 3 nearest regions.
    Real brain: the HCP cortical surface; each point belongs to exactly one Glasser area
    (-1 = medial wall, the inner face where the hemispheres join, which has no area)."""
    anat = config.get("anatomy") or {}
    if anat.get("source", "synthetic") == "hcp":
        from ..hcp import DEFAULT_FILE, load_hcp_brain
        s = load_hcp_brain(anat.get("file", DEFAULT_FILE)).sides[side_name]
        verts, faces = s["verts"].astype(float), s["faces"]
        idx, w, k = s["vertex_region"].astype(int), None, 1
        markers = s["anchors"]  # a point on the surface of each area
    else:
        side = SIDES[side_name]
        verts, faces = anatomy.hemisphere_mesh(side)
        centers = anatomy.region_centers(config["n_regions"], side)
        idx, w = anatomy.vertex_region_weights(verts, centers)
        k, markers = idx.shape[1], centers
    middle = verts.mean(axis=0)  # each model rotates around its own middle
    out = {
        "k": k,
        "v": np.round(_to_three(verts - middle), 4).ravel().tolist(),
        "f": np.asarray(faces).ravel().tolist(),
        "vr": np.asarray(idx).ravel().tolist(),
        "centers": np.round(_to_three(markers - middle), 4).tolist(),
    }
    if w is not None:
        out["vw"] = np.round(w, 3).ravel().tolist()
    return out


def _function_payload(run: dict, side: str, n_regions: int) -> dict:
    """Which function (job) each region belongs to, for the 'show functions' overlay."""
    names = list(run["functions"][side])
    region_function = np.full(n_regions, -1)
    for i, name in enumerate(names):
        region_function[run["functions"][side][name]] = i
    return {"names": names, "region": region_function.tolist()}


def build_viewer(config: dict, results: dict, summary: dict, path: str | Path) -> Path:
    stride = config.get("frame_stride", 4)
    window = config["analysis"]["window"]
    conditions = {}
    for name, res in results.items():
        run, healthy = res["run"], res["healthy"]
        entry = {}
        for side in SIDES:
            entry[side] = _ints(run[side][::stride])
            entry[f"{side}_damage"] = _ints(run[f"{side}_damage"][::stride])
            entry[f"into_{side}"] = _ints(run[f"into_{side}"][::stride])
            score = rolling_function_score(run[side], healthy[side], run[f"{side}_task"], window, stride)
            entry[f"score_{side}"] = [None if s is None else round(s, 3) for s in score]
        conditions[name] = entry
    first = next(iter(results.values()))["run"]
    payload = {
        "meta": {
            "steps": config["steps"], "stride": stride, "window": window,
            "n_frames": len(range(0, config["steps"], stride)),
            "damaged": config["lesion"]["hemisphere"],
            "start_step": config["lesion"]["start_step"],
            "lesion_center": int(first["lesion_center"]),
            "strength": config["corpus_callosum"]["strength"],
            "wiring": config["corpus_callosum"].get("wiring", "mirror"),
            "lesion_type": config["lesion"].get("type", "spreading"),
        },
        "specialization": {s: config[f"{s}_hemisphere"]["specialization"] for s in SIDES},
        "mesh": {s: _mesh_payload(config, s) for s in SIDES},
        "functions": {s: _function_payload(first, s, config["n_regions"]) for s in SIDES},
        "conditions": conditions,
        "summary": summary,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__BRAIN_DATA__", json.dumps(payload, separators=(",", ":")))
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
