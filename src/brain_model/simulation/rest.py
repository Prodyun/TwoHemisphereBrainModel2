"""The brain at rest (no task, no damage), used to compare the model with real resting-state fMRI.

At rest the model's equation (see hemispheres/base.py) reduces to
    x_left  <- (1 - leak_L) x_left  + leak_L tanh(gain_L W_L x_left  + strength C_L x_right + noise_L e)
    x_right <- (1 - leak_R) x_right + leak_R tanh(gain_R W_R x_right + strength C_R x_left  + noise_R e)
This file runs that for MANY settings at once (one row per setting), which makes fitting fast.
tests/test_fit.py checks that it gives exactly the same numbers as the full Brain model.

BOLD stand-in (an assumption): fMRI measures slow blood-flow changes, not neural activity directly.
We smooth activity twice with a slow filter (time constant `tau` steps) before computing
functional connectivity (FC: the correlation between every pair of areas).
"""
import numpy as np

PARAMS = ("leak_l", "gain_l", "noise_l", "leak_r", "gain_r", "noise_r", "strength")


def _route(C: np.ndarray | None, X: np.ndarray) -> np.ndarray:
    """X is (settings, areas). Mirror wiring (None) = same area on the other side."""
    return X if C is None else X @ C.T


def rest_steps(W_l, W_r, C_l, C_r, p: dict, noise: np.ndarray):
    """Yield (x_left, x_right), each (settings, areas), for every time step. noise: (2, steps, areas)."""
    k, n = len(p["gain_l"]), W_l.shape[0]
    col = {name: np.asarray(p[name], dtype=float)[:, None] for name in PARAMS}
    xl, xr = np.zeros((k, n)), np.zeros((k, n))
    for t in range(noise.shape[1]):
        into_l = col["strength"] * _route(C_l, xr)
        into_r = col["strength"] * _route(C_r, xl)
        drive_l = col["gain_l"] * (xl @ W_l.T) + into_l + col["noise_l"] * noise[0, t]
        drive_r = col["gain_r"] * (xr @ W_r.T) + into_r + col["noise_r"] * noise[1, t]
        xl = (1 - col["leak_l"]) * xl + col["leak_l"] * np.tanh(drive_l)
        xr = (1 - col["leak_r"]) * xr + col["leak_r"] * np.tanh(drive_r)
        yield xl, xr


def rest_fc(W_l, W_r, C_l, C_r, p: dict, noise: np.ndarray, tau: float = 10.0,
            burn_in: int = 500, sample_every: int = 5) -> np.ndarray:
    """Simulated functional connectivity, (settings, 2n, 2n), left areas first, as Fisher z."""
    a = 1.0 / tau
    b1 = b2 = None
    samples = []
    for t, (xl, xr) in enumerate(rest_steps(W_l, W_r, C_l, C_r, p, noise)):
        x = np.concatenate([xl, xr], axis=1)
        if b1 is None:
            b1, b2 = x.copy(), x.copy()
        b1 += a * (x - b1)
        b2 += a * (b1 - b2)
        if t >= burn_in and (t - burn_in) % sample_every == 0:
            samples.append(b2.astype(np.float32))
    return correlation_fc(np.stack(samples, axis=1))  # (settings, time, areas)


def correlation_fc(ts: np.ndarray) -> np.ndarray:
    """ts: (..., time, areas) -> Fisher-z correlation matrices (..., areas, areas), diagonal 0."""
    z = ts - ts.mean(axis=-2, keepdims=True)
    sd = z.std(axis=-2, keepdims=True)
    z = np.divide(z, sd, out=np.zeros_like(z), where=sd > 0)
    r = np.matmul(np.swapaxes(z, -1, -2), z) / ts.shape[-2]
    fc = np.arctanh(np.clip(r, -0.999, 0.999))
    idx = np.arange(fc.shape[-1])
    fc[..., idx, idx] = 0.0
    return fc
