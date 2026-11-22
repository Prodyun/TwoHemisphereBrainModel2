"""Download the real-brain files we use from the ENIGMA Toolbox (no login needed).

    python scripts/fetch_enigma_data.py

What we get (all derived from the Human Connectome Project, group level):
  * conte69 32k cortical surfaces (left and right): the real brain shape
  * the Glasser (HCP-MMP1.0) atlas on that surface: 180 areas per hemisphere
  * the HCP group structural connectome in Glasser areas (diffusion MRI): the wiring
  * the HCP group functional connectome in Glasser areas (resting fMRI): used as a check

Files go to data/raw/enigma/ (git-ignored). The download is pinned to one ENIGMA version so
results can be reproduced. Source: Larivière et al. 2021, Nature Methods (ENIGMA Toolbox).
"""
import urllib.request
from pathlib import Path

ENIGMA_COMMIT = "b08974b55243060cbc1fad12c87048037446e8f7"
BASE = f"https://raw.githubusercontent.com/MICA-MNI/ENIGMA/{ENIGMA_COMMIT}/enigmatoolbox/datasets/"
FILES = [
    "surfaces/conte69_32k_lh.gii",
    "surfaces/conte69_32k_rh.gii",
    "surfaces/conte69_32k_lh_mask.csv",
    "surfaces/conte69_32k_rh_mask.csv",
    "parcellations/glasser_360_conte69.csv",
    "matrices/hcp_connectivity/strucMatrix_ctx_glasser_360.csv",
    "matrices/hcp_connectivity/strucLabels_ctx_glasser_360.csv",
    "matrices/hcp_connectivity/funcMatrix_ctx_glasser_360.csv",
    "matrices/hcp_connectivity/funcLabels_ctx_glasser_360.csv",
]
OUT = Path("data/raw/enigma")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = OUT / Path(name).name
        if target.exists():
            print(f"already have {target.name}")
            continue
        print(f"downloading {target.name} ...")
        urllib.request.urlretrieve(BASE + name, target)
    print(f"Done. Files are in {OUT}")


if __name__ == "__main__":
    main()
