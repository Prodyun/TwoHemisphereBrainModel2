"""Real resting-state fMRI -> one time series per Glasser area -> functional connectivity (FC).

HCP stores fMRI as CIFTI "dtseries" files: one column per grayordinate (cortical surface vertex
or subcortical voxel), one row per time point (every 0.72 s). We keep the cortical vertices,
average all vertices of each Glasser area, and correlate every pair of areas.
"""
from pathlib import Path

import numpy as np

from ..simulation.rest import correlation_fc

STRUCTURES = {"left": "CIFTI_STRUCTURE_CORTEX_LEFT", "right": "CIFTI_STRUCTURE_CORTEX_RIGHT"}


def area_timeseries(path: str | Path, vertex_region: dict[str, np.ndarray], n: int) -> np.ndarray:
    """(time, 2n) mean signal of each area, left areas first. vertex_region: per side, area of each
    surface vertex (-1 = none)."""
    import nibabel as nib
    img = nib.load(str(path))
    axis = img.header.get_axis(1)
    data = np.asarray(img.dataobj, dtype=np.float32)  # (time, grayordinates)
    out = np.zeros((data.shape[0], 2 * n), dtype=np.float32)
    found = set()
    for name, slc, bm in axis.iter_structures():
        side = next((s for s, struct in STRUCTURES.items() if struct == name), None)
        if side is None:
            continue
        found.add(side)
        region = vertex_region[side][bm.vertex]
        keep = region >= 0
        cols = data[:, slc][:, keep]
        counts = np.bincount(region[keep], minlength=n)
        sums = np.zeros((data.shape[0], n), dtype=np.float32)
        for r in range(n):  # sum each area's vertices
            if counts[r]:
                sums[:, r] = cols[:, region[keep] == r].sum(axis=1)
        offset = 0 if side == "left" else n
        out[:, offset:offset + n] = sums / np.maximum(counts, 1)
    if found != set(STRUCTURES):
        raise ValueError(f"{path}: expected left and right cortex in the file")
    return out


def subject_fc(run_paths: list[str | Path], vertex_region: dict[str, np.ndarray], n: int) -> np.ndarray:
    """FC (Fisher z) for one subject, averaged over runs."""
    fcs = [correlation_fc(area_timeseries(p, vertex_region, n).astype(float)) for p in run_paths]
    return np.mean(fcs, axis=0)
