"""Fit the resting model to real resting-state fMRI, separately for each hemisphere, with controls.

    python scripts/fit_to_fmri.py            # about 10 minutes on a laptop

Targets: the HCP group FC (always) and each pilot subject's FC (if scripts/process_fmri.py was run).
For each target we fit three versions of the model:
  1. real wiring, separate left/right settings   (the main model)
  2. real wiring, identical left/right settings  (control: do the sides really differ?)
  3. shuffled wiring, separate settings          (control: does the real wiring matter?)
Also reported: how well the wiring alone (no dynamics) matches FC, a common reference point.

Writes results/fit_<time>/fit.json and figures, and prints settings to copy into configs/hcp.yaml.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain_model import anatomy
from brain_model.hcp import load_hcp_brain
from brain_model.simulation.fit import block_scores, fit, rewire_bipartite, rewire_symmetric
from brain_model.simulation.rest import rest_fc
from brain_model.utils import load_config


def segregation(fc: np.ndarray) -> dict[str, float]:
    """Gotts et al. 2013 check: how much more each hemisphere's areas talk to their own side than
    to the other side (higher = more 'segregated'; lower = more 'integrated' across hemispheres)."""
    n = fc.shape[0] // 2
    L, R, LR = fc[:n, :n], fc[n:, n:], fc[:n, n:]
    off = ~np.eye(n, dtype=bool)
    return {"left": float(L[off].mean() - LR.mean()), "right": float(R[off].mean() - LR.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hcp.yaml")
    parser.add_argument("--steps", type=int, default=12000,
                        help="longer runs give more reliable connectivity estimates")
    parser.add_argument("--quick", action="store_true", help="fewer settings (for a fast check)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    brain = load_hcp_brain(cfg["anatomy"]["file"])
    n = brain.n
    wiring = cfg["corpus_callosum"].get("wiring", "real")
    W_l, W_r = anatomy.row_normalize(brain.sc_left), anatomy.row_normalize(brain.sc_right)
    C_l, C_r = anatomy.callosal_wiring(cfg["anatomy"], wiring, n)

    targets = {"group": brain.fc}
    subj_file = Path("data/processed/subject_fc.npz")
    if subj_file.exists():
        z = np.load(subj_file)
        targets.update({f"subject {k}": z[k] for k in z.files})
    print("Fitting to:", ", ".join(targets))

    rng = np.random.default_rng(cfg["seed"])
    noise = rng.normal(size=(2, args.steps, n))
    sizes = dict(n_random=40, n_refine=40) if args.quick else dict(n_random=120, n_refine=80)

    # shuffled-wiring control (same number of links per area, same strengths, random partners)
    sl = rewire_symmetric(brain.sc_left, rng)
    sr = rewire_symmetric(brain.sc_right, rng)
    slr = rewire_bipartite(brain.sc_left_right, rng)
    Ws_l, Ws_r = anatomy.row_normalize(sl), anatomy.row_normalize(sr)
    Cs_l, Cs_r = (None, None) if C_l is None else (anatomy.row_normalize(slr), anatomy.row_normalize(slr.T))

    models = {
        "real wiring, separate L/R settings": (W_l, W_r, C_l, C_r, False),
        "real wiring, identical L/R settings": (W_l, W_r, C_l, C_r, True),
        "shuffled wiring, separate L/R settings": (Ws_l, Ws_r, Cs_l, Cs_r, False),
    }
    results = {}
    for name, (wl, wr, cl, cr, identical) in models.items():
        print(f"\n{name} ...")
        results[name] = fit(wl, wr, cl, cr, targets, noise, rng, identical=identical, **sizes)
        for t, r in results[name].items():
            print(f"  {t:10s} score {r['score']:.3f}  (left {r['r_left']:.2f}, right {r['r_right']:.2f}, "
                  f"between {r['r_between']:.2f})")

    sc_full = np.block([[brain.sc_left, brain.sc_left_right], [brain.sc_left_right.T, brain.sc_right]])
    wiring_alone = {t: block_scores(sc_full, emp) for t, emp in targets.items()}
    print("\nwiring alone (no dynamics):",
          {t: round(s["score"], 3) for t, s in wiring_alone.items()})

    main_name = "real wiring, separate L/R settings"
    best_fc, seg = {}, {}
    for t, emp in targets.items():
        p = {k: np.array([v]) for k, v in results[main_name][t]["params"].items()}
        best_fc[t] = rest_fc(W_l, W_r, C_l, C_r, p, noise)[0]
        seg[t] = {"real": segregation(emp), "model": segregation(best_fc[t])}

    out_dir = Path("results") / datetime.now().strftime("fit_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit.json").write_text(json.dumps(
        {"wiring": wiring, "steps": args.steps, "models": results, "wiring_alone": wiring_alone,
         "segregation": seg}, indent=2), encoding="utf-8")
    make_figures(out_dir, results, wiring_alone, targets, best_fc, main_name)

    print("\nLeft vs right (Gotts et al. 2013: left more segregated, right more integrated):")
    for t, s in seg.items():
        print(f"  {t:10s} real: left {s['real']['left']:.3f}, right {s['real']['right']:.3f}   "
              f"model: left {s['model']['left']:.3f}, right {s['model']['right']:.3f}")

    g = results[main_name]["group"]["params"]
    print("\nSuggested settings for configs/hcp.yaml (fitted to the HCP group):")
    print(f"  left_hemisphere.dynamics:  {{leak: {g['leak_l']:.3f}, gain: {g['gain_l']:.3f}, noise: {g['noise_l']:.3f}}}")
    print(f"  right_hemisphere.dynamics: {{leak: {g['leak_r']:.3f}, gain: {g['gain_r']:.3f}, noise: {g['noise_r']:.3f}}}")
    print(f"  corpus_callosum.strength:  {g['strength']:.3f}")
    print(f"\nSaved {out_dir}")


def make_figures(out_dir, results, wiring_alone, targets, best_fc, main_name) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#2a6cff", "#8aa9e8", "#b0b4bc", "#555b66"]
    names = list(results) + ["wiring alone"]
    tl = list(targets)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [(wiring_alone if name == "wiring alone" else results[name])[t]["score"] for t in tl]
        ax.bar(np.arange(len(tl)) + i * w, vals, w, label=name, color=colors[i % len(colors)])
    ax.set_xticks(np.arange(len(tl)) + w * (len(names) - 1) / 2, tl)
    ax.set_ylabel("match to real fMRI (r)")
    ax.set_title("How well each model reproduces real resting-brain connectivity")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out_dir / "fit_scores.png", dpi=200)
    plt.close(fig)

    emp, sim = targets["group"], best_fc["group"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, m, title in ((axes[0], emp, "Real (HCP group)"), (axes[1], sim, "Model (best fit)")):
        lim = np.percentile(np.abs(m), 98)
        ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title)
        ax.set_xticks([90, 270], ["left", "right"])
        ax.set_yticks([90, 270], ["left", "right"])
        ax.axhline(179.5, color="k", lw=0.6)
        ax.axvline(179.5, color="k", lw=0.6)
    fig.suptitle("Functional connectivity: every pair of the 360 areas")
    fig.tight_layout()
    fig.savefig(out_dir / "fc_real_vs_model.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    for ax, base in zip(axes, ("gain", "leak", "noise")):
        for j, t in enumerate(tl):
            p = results[main_name][t]["params"]
            ax.plot([0, 1], [p[f"{base}_l"], p[f"{base}_r"]], "o-", color=colors[j % 2], alpha=0.8, label=t)
        ax.set_xticks([0, 1], ["left", "right"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(base)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fitted settings: left vs right hemisphere")
    fig.tight_layout()
    fig.savefig(out_dir / "fitted_settings.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
