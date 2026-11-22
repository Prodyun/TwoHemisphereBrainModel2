"""Figures for a patient's plan: where to stimulate (brain maps) and predicted recovery curves."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from ..hcp import load_hcp_brain  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
GAIN_CMAP = LinearSegmentedColormap.from_list("gain", ["#c8412f", "#e9e8e3", "#2a78d6"])
DAMAGE_CMAP = LinearSegmentedColormap.from_list("damage", ["#e9e8e3", "#f0b35a", "#c8412f"])


def _surface(ax, side_data, values, cmap, norm, title):
    v, reg = side_data["verts"], side_data["vertex_region"].astype(int)
    outer = v[:, 0] < np.median(v[:, 0]) if v[:, 0].mean() < 0 else v[:, 0] > np.median(v[:, 0])
    x = -v[outer, 1]
    r = reg[outer]
    colour = np.where(r >= 0, values[np.clip(r, 0, None)], np.nan)
    blank = np.isnan(colour)
    ax.scatter(x[blank], v[outer, 2][blank], s=1.2, color="#bdbcb6", lw=0, rasterized=True)
    sc = ax.scatter(x[~blank], v[outer, 2][~blank], c=colour[~blank], cmap=cmap, norm=norm, s=1.2, lw=0,
                    rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, color=INK)
    return sc


def target_map(rows: list[dict], cfg: dict, damage: dict, out) -> None:
    """Left: the damaged hemisphere (damage). Right: the healthy hemisphere coloured by how much
    stimulating each area improves final function compared with practice alone (best sign)."""
    brain = load_hcp_brain(cfg["anatomy"]["file"])
    damaged = cfg["lesion"]["hemisphere"]
    healthy = "right" if damaged == "left" else "left"
    names = {s: brain.side_names(s) for s in ("left", "right")}
    base = next(r for r in rows if r["kind"] == "none")["final_function"]
    gain = np.full(brain.n, np.nan)
    for r in rows:
        if r["kind"] != "candidate":
            continue
        area = r["targets"].rsplit(" ", 1)[0]
        i = names[healthy].index(area)
        g = (r["final_function"] - base) * 100
        gain[i] = g if np.isnan(gain[i]) else max(gain[i], g)
    dmg = np.zeros(brain.n)
    for area, d in damage.items():
        if area in names[damaged]:
            dmg[names[damaged].index(area)] = d
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sc0 = _surface(axes[0], brain.sides[damaged], np.where(dmg > 0, dmg, np.nan), DAMAGE_CMAP,
                   plt.Normalize(0, 1), f"{damaged.capitalize()} hemisphere: the damage")
    lim = max(0.5, np.nanmax(np.abs(gain)) if np.isfinite(gain).any() else 0.5)
    sc1 = _surface(axes[1], brain.sides[healthy], gain, GAIN_CMAP, TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                   f"{healthy.capitalize()} hemisphere: where stimulation helps")
    fig.colorbar(sc0, ax=axes[0], shrink=0.7).set_label("damage", color=INK2)
    fig.colorbar(sc1, ax=axes[1], shrink=0.7).set_label("final function vs practice alone (points)\n"
                                                         "blue = helps, red = hurts, grey = not tried", color=INK2)
    fig.suptitle(f"{cfg['patient']['name']}: where to help (front of the brain on the left)", x=0.02, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def recovery_curves(curves: dict[str, list[float]], goal: float, title: str, out) -> None:
    colours = ["#2a78d6", "#1f9e6e", "#f0b35a", "#8a63d2", "#c8412f", "#52514e"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (label, ys), c in zip(curves.items(), colours):
        weeks = np.arange(1, len(ys) + 1)
        ax.plot(weeks, np.array(ys) * 100, color=c, lw=2, label=label)
    ax.axhline(goal * 100, color=INK2, ls="--", lw=1)
    ax.text(1, goal * 100 + 1, f"goal {goal * 100:.0f}%", color=INK2, fontsize=8)
    ax.set_xlabel("weeks after the injury", color=INK2)
    ax.set_ylabel("trained job, % of healthy", color=INK2)
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
