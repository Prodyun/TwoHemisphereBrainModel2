"""Loads the prepared real-brain file (data/processed/hcp_brain.npz, see enigma.py)."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_FILE = "data/processed/hcp_brain.npz"


@dataclass(frozen=True)
class HCPBrain:
    names: tuple[str, ...]              # 360 area names: L_... then R_..., same order on both sides
    sc_left: np.ndarray                 # left-left wiring (180 x 180)
    sc_right: np.ndarray                # right-right wiring
    sc_left_right: np.ndarray           # left-right wiring: [i, j] = left area i <-> right area j
    fc: np.ndarray                      # group resting-state functional connectivity (360 x 360)
    sides: dict                         # per side: verts, faces, vertex_region, centers, anchors

    @property
    def n(self) -> int:
        return len(self.names) // 2

    def side_names(self, side: str) -> list[str]:
        return list(self.names[: self.n] if side == "left" else self.names[self.n:])


@lru_cache(maxsize=4)
def load_hcp_brain(path: str | Path = DEFAULT_FILE) -> HCPBrain:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_enigma_data.py  then  python scripts/prepare_hcp.py")
    z = np.load(path)
    sides = {s: {k: z[f"{s}_{k}"] for k in ("verts", "faces", "vertex_region", "centers", "anchors")}
             for s in ("left", "right")}
    return HCPBrain(names=tuple(str(s) for s in z["names"]), sc_left=z["sc_left"], sc_right=z["sc_right"],
                    sc_left_right=z["sc_left_right"], fc=z["fc"], sides=sides)
