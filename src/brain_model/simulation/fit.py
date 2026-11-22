"""Fitting the resting model to real fMRI, and the controls that keep the fit honest.

Score: how similar simulated FC is to real FC, measured separately for
  * left-left connections   (r_left)
  * right-right connections (r_right)
  * left-right connections  (r_between)
each as a Pearson correlation over all area pairs. The fit maximises their average.

Controls:
  * shuffled wiring: the real connections are randomly rewired, keeping how many links each
    area has and the link strengths (degree-preserving rewiring, Maslov & Sneppen 2002).
    If the real wiring matters, it should fit better than this.
  * identical settings: left and right are forced to use the same settings. If the hemispheres
    really behave differently, separate settings should fit better.
"""
import numpy as np

from .rest import PARAMS, rest_fc

RANGES = {"leak": (0.05, 0.6), "gain": (0.2, 1.2), "noise": (0.01, 0.15), "strength": (0.0, 1.0)}


def block_scores(sim: np.ndarray, emp: np.ndarray) -> dict[str, float]:
    n = emp.shape[0] // 2
    iu = np.triu_indices(n, 1)

    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    s = {"r_left": corr(sim[:n, :n][iu], emp[:n, :n][iu]),
         "r_right": corr(sim[n:, n:][iu], emp[n:, n:][iu]),
         "r_between": corr(sim[:n, n:].ravel(), emp[:n, n:].ravel())}
    s["score"] = float(np.mean([s["r_left"], s["r_right"], s["r_between"]]))
    if not np.isfinite(s["score"]):  # e.g. activity froze, so correlations are undefined: rank last
        s = {k: (v if np.isfinite(v) else float("nan")) for k, v in s.items()}
        s["score"] = -1.0
    return s


def sample_params(k: int, rng: np.random.Generator, identical: bool = False) -> dict:
    p = {}
    for name in PARAMS:
        lo, hi = RANGES[name.split("_")[0]]
        p[name] = rng.uniform(lo, hi, k)
    if identical:
        for base in ("leak", "gain", "noise"):
            p[f"{base}_r"] = p[f"{base}_l"].copy()
    return p


def perturb(best: dict, k: int, rng: np.random.Generator, scale: float, identical: bool) -> dict:
    p = {}
    for name in PARAMS:
        lo, hi = RANGES[name.split("_")[0]]
        p[name] = np.clip(best[name] + rng.normal(0, scale * (hi - lo), k), lo, hi)
    if identical:
        for base in ("leak", "gain", "noise"):
            p[f"{base}_r"] = p[f"{base}_l"].copy()
    return p


def fit(W_l, W_r, C_l, C_r, targets: dict[str, np.ndarray], noise: np.ndarray, rng: np.random.Generator,
        n_random: int = 160, n_refine: int = 80, batch: int = 40, identical: bool = False) -> dict:
    """Random search, then a local search around the best setting. One fit per target FC."""
    out = {}
    for name, emp in targets.items():
        tried, scores = [], []

        def run(p):
            for start in range(0, len(p["gain_l"]), batch):
                sub = {k: v[start:start + batch] for k, v in p.items()}
                for i, fc in enumerate(rest_fc(W_l, W_r, C_l, C_r, sub, noise)):
                    tried.append({k: float(v[i]) for k, v in sub.items()})
                    scores.append(block_scores(fc, emp))

        run(sample_params(n_random, rng, identical))
        best = tried[int(np.argmax([s["score"] for s in scores]))]
        run(perturb(best, n_refine, rng, 0.08, identical))
        i = int(np.argmax([s["score"] for s in scores]))
        out[name] = {"params": tried[i], **scores[i], "n_tried": len(tried)}
    return out


def rewire_symmetric(S: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Degree-preserving random rewiring of an undirected weighted network (weights travel with edges)."""
    S = S.copy()
    i, j = np.nonzero(np.triu(S, 1))
    edges = [list(e) for e in zip(i, j)]
    for _ in range(swaps_per_edge * len(edges)):
        e1, e2 = rng.choice(len(edges), 2, replace=False)
        a, b = edges[e1]
        c, d = edges[e2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or S[a, d] or S[c, b]:
            continue
        w1, w2 = S[a, b], S[c, d]
        S[a, b] = S[b, a] = S[c, d] = S[d, c] = 0
        S[a, d] = S[d, a] = w1
        S[c, b] = S[b, c] = w2
        edges[e1], edges[e2] = [a, d], [c, b]
    return S


def rewire_bipartite(S: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Same idea for left-right links (rows = left areas, columns = right areas)."""
    S = S.copy()
    edges = [list(e) for e in zip(*np.nonzero(S))]
    for _ in range(swaps_per_edge * len(edges)):
        e1, e2 = rng.choice(len(edges), 2, replace=False)
        (a, b), (c, d) = edges[e1], edges[e2]
        if a == c or b == d or S[a, d] or S[c, b]:
            continue
        w1, w2 = S[a, b], S[c, d]
        S[a, b] = S[c, d] = 0
        S[a, d], S[c, b] = w1, w2
        edges[e1], edges[e2] = [a, d], [c, b]
    return S
