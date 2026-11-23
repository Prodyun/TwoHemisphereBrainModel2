"""Download the public per-area data used by the AI Help Predictor (no login needed).

    python scripts/fetch_area_data.py               # ~125 MB into data/raw/area/ (git-ignored)
    python scripts/fetch_area_data.py --dest <dir>

Every file is pinned to a fixed version (a git commit or a figshare file id), so results can be
reproduced. Sources (see docs/data.md for citations and licences):
  * Allen Human Brain Atlas gene expression in the Glasser 360 atlas (enigma-extra; Hawrylycz 2012,
    processed for the ENIGMA Toolbox, Lariviere 2021)
  * Cole-Anticevic brain-wide network partition: each Glasser area -> one of 12 networks (Ji 2019)
  * Glasser 2016 area table with the 22 cortical sections (Patterson, Univ. Arizona)
  * Volumetric Glasser atlas in MNI space (Horn 2016, figshare, CC BY 4.0)
  * Receptor / transporter PET density maps (Hansen 2022 Nat Neurosci; netneurolab)
A finished file is never downloaded twice; an interrupted download restarts that file.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

ENIGMA_EXTRA = "https://raw.githubusercontent.com/saratheriver/enigma-extra/8c9df6e0c37c66830360ad553807326035e180ed/ahba/"
CABNP = "https://raw.githubusercontent.com/ColeLab/ColeAnticevicNetPartition/e4ea9fd709ead8616843924b717c6abce62ea05c/"
DPAT = "https://bitbucket.org/dpat/tools/raw/b287b1c5a8d9a18bbe0974dae250c8821029a64f/REF/ATLASES/"
HANSEN = "https://media.githubusercontent.com/media/netneurolab/hansen_receptors/f8b41da92a733f99368c1f39d4959731aa1316c1/data/PET_nifti_images/"
HANSEN_RAW = "https://raw.githubusercontent.com/netneurolab/hansen_receptors/f8b41da92a733f99368c1f39d4959731aa1316c1/data/PET_nifti_images/"

RECEPTORS = {  # file -> what it measures
    "VAChT_feobv_hc18_aghourian_sum.nii": "acetylcholine transporter (cholinergic; VNS gate)",
    "NAT_MRB_hc77_ding.nii.gz": "noradrenaline transporter (noradrenergic; VNS gate)",
    "D1_SCH23390_hc13_kaller.nii": "dopamine D1 receptor",
    "D2_fallypride_hc49_jaworska.nii": "dopamine D2 receptor",
    "DAT_fpcit_hc174_dukart_spect.nii": "dopamine transporter",
    "GABAa-bz_flumazenil_hc16_norgaard.nii": "GABA-A receptor (inhibition)",
    "NMDA_ge179_hc29_galovic.nii.gz": "NMDA receptor (plasticity)",
}

FILES = [  # (local name, [urls to try in order], minimum size in bytes)
    ("allgenes_stable_r0.2_glasser_360.csv.zip", [ENIGMA_EXTRA + "allgenes_stable_r0.2_glasser_360.csv.zip"], 1_000_000),
    ("CABNP_LabelKey.txt", [CABNP + "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"], 10_000),
    ("CABNP_network_labels.txt", [CABNP + "network_labelfile.txt"], 100),
    ("HCP-MMP1_UniqueRegionList.csv", [DPAT + "HCP-MMP1_UniqueRegionList.csv",
                                       "https://bitbucket.org/dpat/tools/raw/master/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv"],
     10_000),
    ("HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz", ["https://ndownloader.figshare.com/files/5594363"], 100_000),
    ("HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt", ["https://ndownloader.figshare.com/files/5534027"], 1_000),
] + [(f, [HANSEN + f, HANSEN_RAW + f], 50_000) for f in RECEPTORS]

HEADERS = {"User-Agent": "brain-model-data-fetch/1.0"}


def fetch(urls: list[str], target: Path, min_size: int) -> str:
    last = None
    for url in urls:
        try:
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r, \
                    open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            size = tmp.stat().st_size
            if size < min_size:  # e.g. a git-LFS pointer instead of the real file
                tmp.unlink()
                last = f"{url}: only {size} bytes"
                continue
            tmp.replace(target)
            return url
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last = f"{url}: {e}"
    raise RuntimeError(f"could not download {target.name} ({last})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw/area")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, urls, min_size in FILES:
        target = dest / name
        if target.exists() and target.stat().st_size >= min_size:
            print(f"already have {name}")
            continue
        try:
            url = fetch(urls, target, min_size)
            print(f"downloaded {name} ({target.stat().st_size / 1e6:.1f} MB) from {url.split('/')[2]}")
        except RuntimeError as e:
            print(f"FAILED {e}")
            failed.append(name)
    print(f"Done. Files are in {dest}" + (f"; {len(failed)} failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
