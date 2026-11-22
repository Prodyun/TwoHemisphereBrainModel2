"""Builds the interactive 3D viewer: one HTML file showing both hemispheres on a black screen.

Two ways to look at the brain (switch at the top of the page):
  * Separate: each hemisphere is its own 3D model with its own camera, rotated on its own.
  * Together: both hemispheres in their real positions in one scene, with the strongest real
              callosal fibres drawn between them and pulses showing the signal each one carries.

The model has 180 areas per hemisphere; the surface (about 32,000 points per side) is only the
canvas they are painted on. To make that visible, the viewer draws each area's outline, a node at
each area, the area's name on hover, and (network view) the real connections between areas.

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
SCALE = 1000          # time series are stored as integers: value * SCALE
LINKS_PER_AREA = 4    # strongest connections per area drawn in network view
CALLOSAL_LINKS = 150  # strongest left-right fibres drawn in the Together view


def _to_three(p: np.ndarray) -> np.ndarray:
    """Model coords (x side, y front, z up) -> Three.js coords (x, y up, z toward viewer)."""
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def _ints(a: np.ndarray, scale: float = SCALE) -> list:
    return np.round(np.asarray(a) * scale).astype(int).tolist()


def _border_vertices(faces: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Vertices on the edge between two areas (a triangle that touches more than one area)."""
    f = np.asarray(faces).reshape(-1, 3)
    r = region[f]
    mixed = (r[:, 0] != r[:, 1]) | (r[:, 1] != r[:, 2])
    border = np.zeros(len(region), dtype=bool)
    border[f[mixed].ravel()] = True
    return border


def _strongest_links(W: np.ndarray, k: int) -> tuple[list, list]:
    """Each area's k strongest connections, without duplicates: ([[i, j], ...], [weight, ...])."""
    sym = (W + W.T) / 2
    pairs = {}
    for i in range(len(sym)):
        for j in np.argsort(sym[i])[::-1][:k]:
            if sym[i, j] > 0 and i != j:
                pairs[(min(i, j), max(i, j))] = float(sym[i, j])
    keys = sorted(pairs)
    return [list(map(int, p)) for p in keys], [round(pairs[p], 4) for p in keys]


def _mesh_payload(config: dict, side_name: str) -> dict:
    """The 3D surface of one hemisphere, which area colours each surface point, and the areas' layout.

    Stand-in: a smooth hand-made surface; each point blends its 3 nearest regions.
    Real brain: the HCP cortical surface; each point belongs to exactly one Glasser area
    (-1 = medial wall, the inner face where the hemispheres join, which has no area)."""
    anat = config.get("anatomy") or {}
    names, border, links, weights = None, None, [], []
    if anat.get("source", "synthetic") == "hcp":
        from ..hcp import DEFAULT_FILE, load_hcp_brain
        brain = load_hcp_brain(anat.get("file", DEFAULT_FILE))
        s = brain.sides[side_name]
        verts, faces = s["verts"].astype(float), s["faces"]
        idx, w, k = s["vertex_region"].astype(int), None, 1
        markers = s["anchors"]  # a point on the surface of each area
        names = brain.side_names(side_name)
        border = _border_vertices(faces, idx)
        sc = brain.sc_left if side_name == "left" else brain.sc_right
        links, weights = _strongest_links(anatomy.row_normalize(sc), LINKS_PER_AREA)
    else:
        side = SIDES[side_name]
        verts, faces = anatomy.hemisphere_mesh(side)
        centers = anatomy.region_centers(config["n_regions"], side)
        idx, w = anatomy.vertex_region_weights(verts, centers)
        k, markers = idx.shape[1], centers
    middle = verts.mean(axis=0)  # in Separate view each model rotates around its own middle
    out = {
        "k": k,
        "v": np.round(_to_three(verts - middle), 4).ravel().tolist(),
        "f": np.asarray(faces).ravel().tolist(),
        "vr": np.asarray(idx).ravel().tolist(),
        "centers": np.round(_to_three(markers - middle), 4).tolist(),
        "offset": np.round(_to_three(middle[None])[0], 4).tolist(),  # where it sits in the real head
        "names": names,
        "links": links,
        "link_w": weights,
    }
    if w is not None:
        out["vw"] = np.round(w, 3).ravel().tolist()
    if border is not None:
        out["border"] = np.flatnonzero(border).tolist()
    return out


def _function_payload(run: dict, side: str, n_regions: int) -> dict:
    """Which function (job) each region belongs to, for the 'show functions' overlay."""
    names = list(run["functions"][side])
    region_function = np.full(n_regions, -1)
    for i, name in enumerate(names):
        region_function[run["functions"][side][name]] = i
    return {"names": names, "region": region_function.tolist()}


def _callosal_matrices(config: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(into_left, into_right) wiring, receiver x sender; the mirror wiring becomes the identity."""
    into_left, into_right = anatomy.callosal_wiring(
        config.get("anatomy"), config["corpus_callosum"].get("wiring", "mirror"), n)
    eye = np.eye(n)
    return (eye if into_left is None else into_left), (eye if into_right is None else into_right)


def _callosal_links(CL: np.ndarray, CR: np.ndarray, k: int) -> list[list[int]]:
    """The k strongest left-right links as [left area, right area]."""
    both = CL + CR.T                      # [left i, right j]
    order = np.argsort(both, axis=None)[::-1]
    links = []
    for flat in order[:k]:
        i, j = np.unravel_index(flat, both.shape)
        if both[i, j] <= 0:
            break
        links.append([int(i), int(j)])
    return links


def _callosal_signals(run: dict, CL: np.ndarray, CR: np.ndarray, links: list, stride: int) -> dict:
    """Per frame: the signal on each drawn fibre (both directions) and how much each area sends."""
    sl = slice(None, None, stride)
    send_l = (1 - run["left_damage"][sl]) * run["left"][sl]     # what a left area puts out
    send_r = (1 - run["right_damage"][sl]) * run["right"][sl]
    g_l, g_r = run["gain_into_left"][sl], run["gain_into_right"][sl]
    li = np.array([p[0] for p in links], dtype=int)
    rj = np.array([p[1] for p in links], dtype=int)
    out = {
        # right area j -> left area i, and left area i -> right area j
        "r2l": _ints(np.abs(g_l[:, li] * CL[li, rj] * send_r[:, rj])) if links else [],
        "l2r": _ints(np.abs(g_r[:, rj] * CR[rj, li] * send_l[:, li])) if links else [],
        # total signal each area sends across (for the cyan "sending" glow)
        "out_left": _ints(np.abs(send_l) * (g_r @ CR)),
        "out_right": _ints(np.abs(send_r) * (g_l @ CL)),
    }
    return out


def run_payload(config: dict, results: dict, summary: dict) -> dict:
    """Everything that changes from one run to the next (not the meshes): the time series of both
    conditions, which side is damaged, where it started, and the scales for the colours."""
    stride = config.get("frame_stride", 4)
    window = config["analysis"]["window"]
    n = config["n_regions"]
    CL, CR = _callosal_matrices(config, n)
    cc_links = _callosal_links(CL, CR, CALLOSAL_LINKS)
    conditions = {}
    for name, res in results.items():
        run, healthy = res["run"], res["healthy"]
        entry = {}
        for side in SIDES:
            entry[side] = _ints(run[side][::stride])
            entry[f"{side}_damage"] = _ints(run[f"{side}_damage"][::stride])
            entry[f"into_{side}"] = _ints(np.abs(run[f"into_{side}"][::stride]))
            score = rolling_function_score(run[side], healthy[side], run[f"{side}_task"], window, stride)
            entry[f"score_{side}"] = [None if s is None else round(s, 3) for s in score]
        entry.update(_callosal_signals(run, CL, CR, cc_links, stride))
        conditions[name] = entry
    first = next(iter(results.values()))["run"]
    con = conditions.get("connected", next(iter(conditions.values())))
    into_all = np.abs(np.concatenate([np.ravel(con["into_left"]), np.ravel(con["into_right"])]))
    out_all = np.concatenate([np.ravel(con["out_left"]), np.ravel(con["out_right"])])
    flux_all = np.concatenate([np.ravel(con["r2l"]), np.ravel(con["l2r"])]) if cc_links else np.ones(1)
    return {
        "meta": {
            "steps": config["steps"], "stride": stride, "window": window, "scale": SCALE,
            "n_frames": len(range(0, config["steps"], stride)),
            "damaged": config["lesion"]["hemisphere"],
            "start_step": config["lesion"]["start_step"],
            "lesion_center": int(first["lesion_center"]),
            "strength": config["corpus_callosum"]["strength"],
            "wiring": config["corpus_callosum"].get("wiring", "mirror"),
            "lesion_type": config["lesion"].get("type", "spreading"),
            # brightness scales: 98th percentile, so a few extreme moments don't wash everything out
            "max_into": float(max(np.percentile(into_all, 98), 1)),
            "max_out": float(max(np.percentile(out_all, 98), 1)),
            "max_flux": float(max(np.percentile(flux_all, 98), 1)),
        },
        "functions": {s: _function_payload(first, s, n) for s in SIDES},
        "callosal_links": cc_links,
        "conditions": conditions,
        "summary": summary,
    }


def viewer_payload(config: dict, results: dict, summary: dict, interactive: bool = False) -> dict:
    """Run data + the 3D meshes. `interactive` = served by scripts/explore.py, so areas can be picked."""
    payload = run_payload(config, results, summary)
    payload["meta"]["interactive"] = interactive
    payload["specialization"] = {s: config[f"{s}_hemisphere"]["specialization"] for s in SIDES}
    payload["mesh"] = {s: _mesh_payload(config, s) for s in SIDES}
    return payload


def viewer_html(payload: dict) -> str:
    return TEMPLATE.read_text(encoding="utf-8").replace(
        "__BRAIN_DATA__", json.dumps(payload, separators=(",", ":")))


def build_viewer(config: dict, results: dict, summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(viewer_html(viewer_payload(config, results, summary)), encoding="utf-8")
    return path
