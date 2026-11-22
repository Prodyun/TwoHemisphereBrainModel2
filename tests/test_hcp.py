"""Tests for the real-brain (HCP) mode.

They build a tiny fake brain in ENIGMA's file format (6 areas per hemisphere), so they run
without the real downloads. Tests that need the real data are skipped if it is missing.
"""
import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from brain_model import Brain
from brain_model.hcp import load_hcp_brain
from brain_model.hcp.enigma import build_brain_file, connection_strength, vertex_regions
from brain_model.simulation import run_experiment, summarize
from brain_model.utils import load_config
from brain_model.visualization import build_viewer

AREAS = ["V1", "44", "45", "FFC", "4", "7PC"]
N = len(AREAS)
REAL_FILE = Path("data/processed/hcp_brain.npz")


def _write_surface(path: Path, side: int, rng) -> int:
    """Each area gets a small cluster of 4 points; plus 2 medial-wall points. Returns vertex count."""
    pts = []
    for a in range(N):
        centre = np.array([side * 40.0, -60 + 25 * a, 10.0 * (a % 2)])
        pts.append(centre + rng.normal(0, 2, size=(4, 3)))
    pts.append(np.array([[side * 2.0, 0, 0], [side * 2.0, 5, 5]]))
    verts = np.vstack(pts).astype(np.float32)
    faces = np.array([[i, i + 1, i + 2] for i in range(len(verts) - 2)], dtype=np.int32)
    img = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(verts, intent="NIFTI_INTENT_POINTSET"),
        nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE")])
    nib.save(img, str(path))
    return len(verts)


def make_fake_enigma(raw: Path) -> None:
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    names = [f"L_{a}" for a in AREAS] + [f"R_{a}" for a in AREAS]
    (raw / "strucLabels_ctx_glasser_360.csv").write_text(",".join(names))
    sc = np.zeros((2 * N, 2 * N))
    for i in range(N - 1):  # a chain inside each hemisphere, with different strengths left vs right
        sc[i, i + 1] = sc[i + 1, i] = 5.0 + i
        sc[N + i, N + i + 1] = sc[N + i + 1, N + i] = 9.0 - i
    sc[0, N + 0] = sc[N + 0, 0] = 4.0      # callosal fibre: left V1 <-> right V1
    sc[2, N + 1] = sc[N + 1, 2] = -0.5     # left 45 <-> right 44 (slightly negative log weight)
    np.savetxt(raw / "strucMatrix_ctx_glasser_360.csv", sc, delimiter=",")
    np.savetxt(raw / "funcMatrix_ctx_glasser_360.csv", np.abs(rng.normal(size=(2 * N, 2 * N))), delimiter=",")
    nl = _write_surface(raw / "conte69_32k_lh.gii", -1, rng)
    nr = _write_surface(raw / "conte69_32k_rh.gii", 1, rng)
    left = np.r_[np.repeat(np.arange(1, N + 1), 4), [0, 0]]
    right = np.r_[np.repeat(np.arange(N + 1, 2 * N + 1), 4), [0, N]]  # last: a stray left number
    assert len(left) == nl and len(right) == nr
    np.savetxt(raw / "glasser_360_conte69.csv", np.r_[left, right], fmt="%d")


@pytest.fixture(scope="module")
def fake_file(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("hcp")
    make_fake_enigma(root / "raw")
    return build_brain_file(root / "raw", root / "hcp_brain.npz")


def fake_config(path: Path, **changes) -> dict:
    cfg = copy.deepcopy(load_config("configs/hcp.yaml"))
    cfg.update(n_regions=N, steps=240)
    cfg["anatomy"]["file"] = str(path)
    cfg["lesion"].update(region="L_45", start_step=20, spread_rate=0.03)
    cfg["left_hemisphere"]["functions"] = {
        "language": {"areas": ["44", "45"], "pattern": "sequence"},
        "vision": {"areas": ["V1"], "pattern": "sine", "params": {"period": 40}}}
    cfg["right_hemisphere"]["functions"] = {
        "faces": {"areas": ["FFC"], "pattern": "bursts"},
        "spatial_attention": {"areas": ["7PC"], "pattern": "wave"},
        "vision": {"areas": ["V1"], "pattern": "sine", "params": {"period": 40, "phase": 1.0}}}
    for key, value in changes.items():
        cfg[key] = {**cfg[key], **value} if isinstance(value, dict) else value
    return cfg


# ---------- preparing the real-brain file ----------

def test_connection_strength_is_positive_only_where_connected():
    raw = np.array([[0, -1.0, 3.0], [-1.0, 0, 0], [3.0, 0, 0]])
    w = connection_strength(raw)
    assert (w[raw != 0] > 0).all() and (w[raw == 0] == 0).all()
    assert w[0, 2] > w[0, 1]  # order is kept


def test_right_hemisphere_labels_are_mapped_and_stray_labels_dropped():
    assert vertex_regions(np.array([0, 7, 8, 13, 6]), first=7, n=6).tolist() == [-1, 0, 1, -1, -1]


def test_brain_file_has_separate_left_and_right_wiring(fake_file):
    b = load_hcp_brain(fake_file)
    assert b.n == N and b.side_names("right")[1] == "R_44"
    assert not np.allclose(b.sc_left, b.sc_right)
    assert b.sc_left_right[0, 0] > 0 and b.sc_left_right[2, 1] > 0
    assert (b.sides["left"]["vertex_region"] == -1).sum() == 2
    assert (b.sides["right"]["vertex_region"] == -1).sum() == 2  # medial + stray
    assert (b.sides["left"]["centers"][:, 0] < 0).all() and (b.sides["right"]["centers"][:, 0] > 0).all()


# ---------- model with the real brain ----------

def test_hcp_hemispheres_use_their_own_wiring(fake_file):
    brain = Brain(fake_config(fake_file))
    assert np.allclose(brain.left.W.sum(axis=1), 1) and np.allclose(brain.right.W.sum(axis=1), 1)
    assert not np.allclose(brain.left.W, brain.right.W)


def test_functions_only_drive_their_own_areas(fake_file):
    brain = Brain(fake_config(fake_file))
    parts = brain.left.task_parts(5)
    assert set(parts) == {"language", "vision"}
    lang = brain.left.functions[0].regions
    outside = np.setdiff1d(np.arange(N), lang)
    assert not parts["language"][outside].any()
    total = brain.left.task_input(5)
    assert not total[[brain.left.area_index("FFC"), brain.left.area_index("4")]].any()


def test_function_areas_must_not_overlap(fake_file):
    cfg = fake_config(fake_file)
    cfg["left_hemisphere"]["functions"]["vision"]["areas"] = ["V1", "44"]
    with pytest.raises(ValueError):
        Brain(cfg)


def test_unknown_area_rejected(fake_file):
    with pytest.raises(ValueError):
        Brain(fake_config(fake_file, lesion={"region": "L_NOPE"}))
    with pytest.raises(ValueError):
        Brain(fake_config(fake_file, lesion={"region": "R_44"}))  # wrong hemisphere


def test_functions_need_the_real_brain():
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg["left_hemisphere"]["functions"] = {"language": {"areas": ["44"], "pattern": "sequence"}}
    with pytest.raises(ValueError):
        Brain(cfg)


def test_per_hemisphere_settings_override_shared_ones(fake_file):
    cfg = fake_config(fake_file, left_hemisphere={"dynamics": {"gain": 0.3, "noise": 0.01}},
                      right_hemisphere={"dynamics": {}})
    brain = Brain(cfg)
    assert brain.left.gain == 0.3 and brain.left.noise == 0.01
    shared = cfg["dynamics"]
    assert brain.right.gain == shared["gain"] and brain.right.noise == shared["noise"]


# ---------- callosum wiring ----------

def test_real_callosum_follows_measured_fibres(fake_file):
    brain = Brain(fake_config(fake_file))
    brain.right.x[:] = 0
    brain.right.x[1] = 1.0  # right 44 is active
    into_left, _ = brain.callosum.signals(brain.left, brain.right)
    # right 44 is wired to left 45 (not to left 44)
    assert into_left[2] > 0 and np.count_nonzero(into_left) == 1


def test_mirror_callosum_links_matching_areas(fake_file):
    brain = Brain(fake_config(fake_file, corpus_callosum={"wiring": "mirror"}))
    brain.right.x[:] = 0
    brain.right.x[1] = 1.0
    into_left, _ = brain.callosum.signals(brain.left, brain.right)
    assert into_left[1] > 0 and np.count_nonzero(into_left) == 1


def test_both_wiring_adds_mirror_links(fake_file):
    brain = Brain(fake_config(fake_file, corpus_callosum={"wiring": "both"}))
    brain.right.x[:] = 0
    brain.right.x[1] = 1.0
    into_left, _ = brain.callosum.signals(brain.left, brain.right)
    assert into_left[1] > 0 and into_left[2] > 0


def test_real_wiring_needs_real_brain():
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg["corpus_callosum"]["wiring"] = "real"
    with pytest.raises(ValueError):
        Brain(cfg)


# ---------- lesions ----------

def _run_lesion(cfg, steps):
    brain = Brain(cfg)
    for t in range(steps):
        brain.step(t, np.zeros(N), np.zeros(N))
    return brain


def test_lesion_starts_at_named_area(fake_file):
    cfg = fake_config(fake_file)
    brain = _run_lesion(cfg, cfg["lesion"]["start_step"] + 1)
    assert brain.lesion.center_region == 2 and brain.left.damage[2] > 0


def test_stroke_hits_once_and_does_not_spread(fake_file):
    cfg = fake_config(fake_file, lesion={"type": "stroke", "stroke_neighbors": 1})
    start = cfg["lesion"]["start_step"]
    brain = _run_lesion(cfg, start + 1)
    after_hit = brain.left.damage.copy()
    assert np.count_nonzero(after_hit) == 2  # the area + its strongest neighbour
    for t in range(start + 1, start + 100):
        brain.step(t, np.zeros(N), np.zeros(N))
    assert np.allclose(brain.left.damage, after_hit)


def test_spreading_lesion_spreads(fake_file):
    cfg = fake_config(fake_file)
    brain = _run_lesion(cfg, cfg["lesion"]["start_step"] + 150)
    assert np.count_nonzero(brain.left.damage) > 2


def test_cross_spread_follows_callosal_wiring(fake_file):
    cfg = fake_config(fake_file, lesion={"cross_spread": 1.0})
    brain = _run_lesion(cfg, cfg["lesion"]["start_step"] + 60)
    # left 45 is wired to right 44, so damage leaks there (and not to right 45 directly)
    assert brain.right.damage[1] > 0


def test_invalid_lesion_type_rejected(fake_file):
    with pytest.raises(ValueError):
        Brain(fake_config(fake_file, lesion={"type": "flu"}))


# ---------- experiment, scoring, viewer ----------

def test_experiment_reports_each_function(fake_file, tmp_path):
    cfg = fake_config(fake_file)
    results = run_experiment(cfg)
    summary = summarize(results, cfg)
    assert set(summary["function_benefit"]) == {"language", "vision"}
    assert set(summary["split"]["functions"]["right"]) == {"faces", "spatial_attention", "vision"}
    for f, score in summary["split"]["functions"]["right"].items():
        assert score == pytest.approx(1.0), f  # split brain: the healthy side is untouched
    html = build_viewer(cfg, results, summary, tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert '"k":1' in html and '"functions"' in html


# ---------- the real downloaded brain (skipped if not prepared) ----------

@pytest.mark.skipif(not REAL_FILE.exists(), reason="run scripts/fetch_enigma_data.py and prepare_hcp.py")
def test_real_brain_config_builds():
    cfg = load_config("configs/hcp.yaml")
    brain = Brain(cfg)
    assert brain.left.region_names[brain.lesion.center_region] == "L_44"
    names = {h.name: [f.name for f in h.functions] for h in (brain.left, brain.right)}
    assert "language" in names["left"] and "faces" in names["right"]
    assert "vision" in names["left"] and "vision" in names["right"]  # shared job
    assert not np.allclose(brain.left.W, brain.right.W)


def test_viewer_shows_areas_wiring_and_callosal_fibres(fake_file, tmp_path):
    import json
    import re
    cfg = fake_config(fake_file)
    results = run_experiment(cfg)
    html = build_viewer(cfg, results, summarize(results, cfg), tmp_path / "viewer.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script id="brain-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    left = data["mesh"]["left"]
    assert left["names"] and len(left["centers"]) == N          # one node per area, with its name
    assert left["border"]                                       # area outlines
    assert all(0 <= i < N and 0 <= j < N for i, j in left["links"])
    frames = data["meta"]["n_frames"]
    con = data["conditions"]["connected"]
    assert data["callosal_links"] and len(con["r2l"]) == frames
    assert len(con["r2l"][0]) == len(data["callosal_links"])
    assert max(map(max, con["r2l"])) > 0                       # signals cross when connected
    assert max(map(max, data["conditions"]["split"]["r2l"])) == 0  # and not when cut
