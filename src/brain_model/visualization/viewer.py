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


def _mesh_payload(side: int, n_regions: int) -> dict:
    verts, faces = anatomy.hemisphere_mesh(side)
    centers = anatomy.region_centers(n_regions, side)
    idx, w = anatomy.vertex_region_weights(verts, centers)
    middle = verts.mean(axis=0)  # each model rotates around its own middle
    return {
        "v": np.round(_to_three(verts - middle), 4).ravel().tolist(),
        "f": faces.ravel().tolist(),
        "vr": idx.ravel().tolist(),
        "vw": np.round(w, 3).ravel().tolist(),
        "centers": np.round(_to_three(centers - middle), 4).tolist(),
    }


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
        },
        "specialization": {s: config[f"{s}_hemisphere"]["specialization"] for s in SIDES},
        "mesh": {s: _mesh_payload(v, config["n_regions"]) for s, v in SIDES.items()},
        "conditions": conditions,
        "summary": summary,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__BRAIN_DATA__", json.dumps(payload, separators=(",", ":")))
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
