"""Hemisphere shape, brain regions and their wiring (structural connectivity).

This is a stand-in until real HCP data is plugged in:
  * the shape is a smooth half-brain, not a real cortical surface (HCP: *.L/R.midthickness.surf.gii)
  * the wiring is built from distances between regions (HCP: a diffusion-MRI connectome)

Coordinates: x = left(-)/right(+), y = back(-)/front(+), z = down(-)/up(+).
Everything is built for the LEFT hemisphere and mirrored (x -> -x) for the right, so region i
on the left and region i on the right are mirror partners ("homotopic" regions). The corpus
callosum mainly links these partners.
"""
import numpy as np

RADII = np.array([0.42, 0.85, 0.60])  # half-width, half-length, half-height of one hemisphere
MEDIAL_SQUASH = 0.2                    # flattens the inner (medial) face that faces the other side
MEDIAL_GAP = MEDIAL_SQUASH * RADII[0] + 0.02  # shift so the medial face stays just off the midline

# Fixed random directions for the fold pattern (same every run, same on both sides).
_FOLD_DIRS = np.random.default_rng(7).normal(size=(7, 3))
_FOLD_DIRS /= np.linalg.norm(_FOLD_DIRS, axis=1, keepdims=True)
_FOLD_PHASES = np.random.default_rng(8).uniform(0, 2 * np.pi, 7)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _surface_point(u: np.ndarray, side: int) -> np.ndarray:
    """Map unit directions u (N x 3) onto the hemisphere surface. side = -1 left, +1 right."""
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    blunt = np.sign(u) * np.abs(u) ** 0.8      # rounder, less pointy than a plain ellipsoid
    x, y, z = (blunt * RADII).T.copy()

    back = _smoothstep(-uy, 0.2, 1.0)          # occipital lobe: lower and narrower at the back
    z *= 1 - 0.2 * back
    x *= 1 - 0.15 * back
    front = _smoothstep(uy, 0.3, 1.0)          # frontal lobe: tall and rounded
    z *= 1 + 0.08 * front
    z = np.where(z < 0, 0.55 * z, z)           # flat base
    lateral = _smoothstep(-ux, -0.1, 0.5)
    temporal = np.exp(-(((uy - 0.12) / 0.32) ** 2 + ((uz + 0.6) / 0.3) ** 2)) * lateral
    z -= 0.17 * temporal                        # temporal lobe hangs below the middle-front
    x -= 0.05 * temporal

    # Sylvian fissure: a deep groove separating the temporal lobe, rising toward the back.
    a, b = np.array([0.5, -0.12]), np.array([-0.3, 0.16])
    yz = np.column_stack([y, z])
    s = np.clip(((yz - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    dist = np.linalg.norm(yz - (a + s[:, None] * (b - a)), axis=1)
    groove = 0.06 * np.exp(-(dist / 0.035) ** 2) * lateral

    # Folds (sulci): grooves along the zero-lines of a smooth random pattern.
    g = np.sin(15 * u @ _FOLD_DIRS.T + _FOLD_PHASES).sum(axis=1) / np.sqrt(len(_FOLD_PHASES))
    folds = 0.045 * np.exp(-(g / 0.25) ** 2)

    p = np.column_stack([x, y, z]) * (1 - groove - folds)[:, None]
    medial = p[:, 0] > 0                       # left hemisphere: medial side is x > 0
    p[medial, 0] *= MEDIAL_SQUASH
    p[:, 0] -= MEDIAL_GAP
    if side == 1:
        p[:, 0] *= -1
    return p


def fibonacci_directions(n: int) -> np.ndarray:
    """n evenly spread unit vectors on a sphere."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    rho = np.sqrt(1 - z ** 2)
    return np.column_stack([rho * np.cos(phi), z, rho * np.sin(phi)])  # y gets the poles' axis


def region_centers(n_regions: int, side: int) -> np.ndarray:
    return _surface_point(fibonacci_directions(n_regions), side)


def hemisphere_mesh(side: int, n_lat: int = 96, n_lon: int = 144) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh of the hemisphere surface: (vertices V x 3, faces F x 3)."""
    lat = np.linspace(0, np.pi, n_lat + 1)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    th, ph = np.meshgrid(lat, lon, indexing="ij")
    # Poles along y (front/back), so the long axis of the brain is smooth.
    u = np.column_stack([
        (np.sin(th) * np.cos(ph)).ravel(),
        np.cos(th).ravel(),
        (np.sin(th) * np.sin(ph)).ravel(),
    ])
    verts = _surface_point(u, side)
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            faces += [[a, c, b], [b, c, d]]
    faces = np.array(faces)
    if side == 1:  # mirroring flips triangle orientation; flip back so normals point outward
        faces = faces[:, [0, 2, 1]]
    return verts, faces


def structural_connectivity(centers: np.ndarray, neighbors: int, long_range_fraction: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Connection matrix W (regions x regions). Row i says how much region i listens to each other region.

    Each region links to its nearest neighbours (stronger when closer), plus some random
    long-distance links. Rows are normalised to sum to 1 so activity stays stable.
    """
    n = len(centers)
    dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nearest = np.argsort(dist, axis=1)[:, :neighbors]
    scale = np.mean(dist[np.arange(n)[:, None], nearest])
    W = np.zeros((n, n))
    for i in range(n):
        W[i, nearest[i]] = np.exp(-dist[i, nearest[i]] / scale)
    n_long = int(round(long_range_fraction * n * neighbors))
    src, dst = rng.integers(0, n, n_long), rng.integers(0, n, n_long)
    keep = src != dst
    W[src[keep], dst[keep]] = np.maximum(W[src[keep], dst[keep]], 0.5)
    W = np.maximum(W, W.T)  # brain wiring here is two-way
    return W / W.sum(axis=1, keepdims=True)


def row_normalize(W: np.ndarray) -> np.ndarray:
    """Scale each row to sum to 1 (rows with no connections stay all zero)."""
    s = W.sum(axis=1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W, dtype=float), where=s > 0)


def _source(anatomy_cfg: dict | None) -> str:
    source = (anatomy_cfg or {}).get("source", "synthetic")
    if source not in ("synthetic", "hcp"):
        raise ValueError("anatomy.source must be 'synthetic' or 'hcp'")
    return source


def _hcp(anatomy_cfg: dict):
    from .hcp import DEFAULT_FILE, load_hcp_brain
    return load_hcp_brain(anatomy_cfg.get("file", DEFAULT_FILE))


def load_hemisphere(side_name: str, n_regions: int, hemi_cfg: dict, anatomy_cfg: dict | None,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """(area centres, wiring W, area names) for one hemisphere.

    synthetic: the hand-made stand-in (mirror-image shapes, distance-based wiring, no names).
    hcp:       the real brain. Each hemisphere has its OWN measured wiring (diffusion MRI), so the
               left and right networks differ the way they do in real people."""
    side = -1 if side_name == "left" else 1
    if _source(anatomy_cfg) == "synthetic":
        centers = region_centers(n_regions, side)
        W = structural_connectivity(centers, hemi_cfg["neighbors"], hemi_cfg["long_range_fraction"], rng)
        return centers, W, None
    brain = _hcp(anatomy_cfg)
    if n_regions != brain.n:
        raise ValueError(f"n_regions must be {brain.n} for the real brain (Glasser atlas)")
    sc = brain.sc_left if side_name == "left" else brain.sc_right
    return brain.sides[side_name]["centers"].copy(), row_normalize(sc), brain.side_names(side_name)


def callosal_wiring(anatomy_cfg: dict | None, wiring: str, n_regions: int
                    ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Who sends callosal signals to whom: (into_left, into_right), each receiver x sender.

    mirror: area i only talks to area i on the other side (None = this simple one-to-one link).
    real:   the measured left-right fibres from diffusion MRI (an area can reach several areas,
            and some areas have no detected direct link).
    both:   real fibres plus a mirror link for every area."""
    if wiring not in ("mirror", "real", "both"):
        raise ValueError("corpus_callosum.wiring must be 'mirror', 'real' or 'both'")
    if wiring == "mirror":
        return None, None
    if _source(anatomy_cfg) != "hcp":
        raise ValueError("corpus_callosum.wiring 'real'/'both' needs anatomy.source: hcp")
    lr = _hcp(anatomy_cfg).sc_left_right
    if wiring == "both":
        lr = lr + np.eye(n_regions) * lr[lr > 0].mean()
    return row_normalize(lr), row_normalize(lr.T)


def vertex_region_weights(verts: np.ndarray, centers: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """For colouring the mesh: each vertex blends its k nearest regions (closer = more weight)."""
    dist = np.linalg.norm(verts[:, None] - centers[None], axis=-1)
    idx = np.argsort(dist, axis=1)[:, :k]
    d = np.take_along_axis(dist, idx, axis=1)
    w = 1 / (d + 1e-3) ** 2
    return idx, w / w.sum(axis=1, keepdims=True)
