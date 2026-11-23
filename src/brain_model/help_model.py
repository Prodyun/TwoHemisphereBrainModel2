"""The AI HELP PREDICTOR: for any damaged area, the chance that the healthy hemisphere can help it
recover, how much, with which practice mode, and through which helper areas.

Model: gradient-boosted decision trees (scikit-learn HistGradientBoosting), trained on thousands
of simulated patients (scripts/build_help_dataset.py). It learns which REAL properties of an area
make help likely:
  wiring      direct callosal fibres in/out (count, strength), a fibre to the mirror area, two-step
              reach through the healthy side's own wiring, same-side connections (HCP diffusion MRI)
  syncing     resting-state functional connectivity with the mirror area (HCP fMRI)
  role        the area's network (Cole-Anticevic), Glasser section, volume, its job in the model
  biology     plasticity / vulnerability genes (Allen atlas) and receptor densities (PET, Hansen 2022)
  the case    severity, disorder, practice mode
Testing is honest: whole AREAS are held out (grouped cross-validation), so every reported score is
for areas the model never saw in training. Compared with baselines (always-average; wiring only).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hcp import load_hcp_brain

ATLAS_FILE = Path("data/processed/area_atlas.npz")
MODEL_FILE = Path("data/processed/help_model.joblib")
MODES = ["physical", "imagery", "bci", "bci+vns", "imagery+sleep"]
PRIMARY = {"4", "3a", "3b", "1", "2", "V1", "A1"}


@dataclass
class AreaFeatures:
    names: list[str]
    columns: list[str]
    table: np.ndarray                     # (360, n features)
    plain: dict = field(default_factory=dict)  # column -> plain-language description


def area_features(anatomy_file: str = "data/processed/hcp_brain.npz", atlas_file: Path = ATLAS_FILE) -> AreaFeatures:
    b = load_hcp_brain(anatomy_file)
    n = b.n
    names = list(b.names)
    lr = b.sc_left_right                                   # [left i, right j]
    cols, plain, data = [], {}, []

    def add(name, values, text):
        cols.append(name)
        plain[name] = text
        data.append(np.asarray(values, float))

    for side in ("left", "right"):
        pass
    cc_in = np.r_[(lr > 0).sum(1), (lr > 0).sum(0)]
    cc_w = np.r_[lr.sum(1), lr.sum(0)]
    mirror = np.r_[np.diag(lr) > 0, np.diag(lr) > 0]
    homo_fc = np.r_[[b.fc[i, n + i] for i in range(n)], [b.fc[n + i, i] for i in range(n)]]
    sc_same = [b.sc_left, b.sc_right]
    deg = np.r_[(b.sc_left > 0).sum(1), (b.sc_right > 0).sum(1)]
    strength = np.r_[b.sc_left.sum(1), b.sc_right.sum(1)]
    # two-step reach: fibres from the healthy side into this area's same-side neighbours
    two_left = (sc_same[0] > 0).astype(float) @ (lr > 0).sum(1)        # left area <- right via left neighbours
    two_right = (sc_same[1] > 0).astype(float) @ (lr > 0).sum(0)
    centers = np.vstack([b.sides["left"]["centers"], b.sides["right"]["centers"]])
    add("cc_fibres", cc_in, "number of direct fibres to the other hemisphere")
    add("cc_strength", cc_w, "strength of its direct fibres across")
    add("mirror_fibre", mirror, "has a measured fibre to its mirror area")
    add("homotopic_fc", homo_fc, "how strongly it syncs with its mirror area at rest")
    add("two_step_reach", np.r_[two_left, two_right], "fibres across reachable through its neighbours")
    add("same_side_links", deg, "number of connections on its own side")
    add("same_side_strength", strength, "strength of its own-side connections")
    add("x_abs", np.abs(centers[:, 0]), "distance from the midline")
    add("y", centers[:, 1], "front-back position")
    add("z", centers[:, 2], "up-down position")
    add("primary", [nm[2:] in PRIMARY for nm in names], "primary motor / touch / visual / hearing area")
    add("right_side", [nm.startswith("R_") for nm in names], "in the right hemisphere")
    if Path(atlas_file).exists():
        a = np.load(atlas_file, allow_pickle=False)
        assert list(a["names"]) == names, "area_atlas.npz order does not match the brain file"
        for net in a["network_names"]:
            add(f"net:{net}", a["network"] == net, f"belongs to the {net} network")
        add("volume", a["volume"], "area size (mm3)")
        for j, g in enumerate(a["gene_names"]):
            add(f"gene:{g}", a["genes"][:, j], f"{g} gene activity")
        for j, r in enumerate(a["receptor_names"]):
            add(f"receptor:{r}", a["receptors"][:, j], f"{r} receptor density")
    return AreaFeatures(names, cols, np.column_stack(data), plain)


CASE_COLUMNS = ["severity", "tumor"] + [f"mode:{m}" for m in MODES]


def case_vector(severity: float, disorder: str, mode: str) -> np.ndarray:
    return np.r_[severity, disorder == "tumor", [mode == m for m in MODES]].astype(float)


def design(rows: list[dict], feats: AreaFeatures) -> tuple[np.ndarray, list[str]]:
    index = {nm: i for i, nm in enumerate(feats.names)}
    X = np.array([np.r_[feats.table[index[r["area"]]], case_vector(r["severity"], r["disorder"], r["mode"])]
                  for r in rows])
    return X, feats.columns + CASE_COLUMNS


def train(rows: list[dict], feats: AreaFeatures, folds: int = 5, seed: int = 0, max_iter: int = 300) -> dict:
    """Fit the models and score them on held-out AREAS. Returns everything predict() needs."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows = [r for r in rows if not r.get("never_impaired")]          # nothing to recover
    X, cols = design(rows, feats)
    groups = np.array([r["area"] for r in rows])
    y_success = np.array([r["success"] for r in rows], int)
    y_help = np.array([r["help"] for r in rows]) * 100
    y_final = np.array([r["final_function"] for r in rows]) * 100
    clf = lambda: HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, random_state=seed)  # noqa: E731
    reg = lambda: HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.05, random_state=seed)  # noqa: E731
    wiring_cols = [cols.index(c) for c in ("cc_fibres", "cc_strength", "homotopic_fc")] + \
        [cols.index(c) for c in CASE_COLUMNS]
    oof = {k: np.zeros(len(rows)) for k in ("p", "p_wiring", "help", "help_wiring", "final")}
    for tr, te in GroupKFold(n_splits=folds).split(X, groups=groups):
        oof["p"][te] = clf().fit(X[tr], y_success[tr]).predict_proba(X[te])[:, 1]
        oof["p_wiring"][te] = clf().fit(X[tr][:, wiring_cols], y_success[tr]).predict_proba(X[te][:, wiring_cols])[:, 1]
        oof["help"][te] = reg().fit(X[tr], y_help[tr]).predict(X[te])
        oof["help_wiring"][te] = reg().fit(X[tr][:, wiring_cols], y_help[tr]).predict(X[te][:, wiring_cols])
        oof["final"][te] = reg().fit(X[tr], y_final[tr]).predict(X[te])
    metrics = {
        "rows": len(rows), "areas": int(len(set(groups))), "success_rate": float(y_success.mean()),
        "auc_success": float(roc_auc_score(y_success, oof["p"])) if len(set(y_success)) > 1 else None,
        "auc_success_wiring_only": float(roc_auc_score(y_success, oof["p_wiring"])) if len(set(y_success)) > 1 else None,
        "r2_help": float(r2_score(y_help, oof["help"])),
        "r2_help_wiring_only": float(r2_score(y_help, oof["help_wiring"])),
        "mae_help_points": float(mean_absolute_error(y_help, oof["help"])),
        "mae_help_always_average": float(mean_absolute_error(y_help, np.full_like(y_help, y_help.mean()))),
        "r2_final": float(r2_score(y_final, oof["final"])),
    }
    models = {"success": clf().fit(X, y_success), "help": reg().fit(X, y_help), "final": reg().fit(X, y_final)}
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(rows), size=min(2000, len(rows)), replace=False)
    imp = permutation_importance(models["help"], X[sample], y_help[sample], n_repeats=5, random_state=seed)
    importance = sorted(zip(cols, imp.importances_mean), key=lambda kv: -kv[1])
    helpers = {}
    for r in rows:
        if r.get("helpers"):
            helpers.setdefault((r["area"], r["disorder"]), {})[r["severity"]] = r["helpers"]
    return {"models": models, "columns": cols, "metrics": metrics, "importance": importance,
            "oof": {"y_help": y_help, "help": oof["help"], "y_success": y_success, "p": oof["p"],
                    "area": groups, "mode": np.array([r["mode"] for r in rows])},
            "helpers": helpers, "feature_names": feats.columns}


class HelpPredictor:
    """Loads the trained models; answers instantly for any area."""

    def __init__(self, bundle: dict, feats: AreaFeatures):
        self.b, self.f = bundle, feats
        self.index = {nm: i for i, nm in enumerate(feats.names)}

    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> "HelpPredictor":
        import joblib
        return cls(joblib.load(path), area_features())

    def predict(self, area: str, severity: float = 0.9, disorder: str = "stroke") -> dict:
        if area not in self.index:
            raise ValueError(f"unknown area {area}")
        a = self.f.table[self.index[area]]
        X = np.array([np.r_[a, case_vector(severity, disorder, m)] for m in MODES])
        m = self.b["models"]
        p = m["success"].predict_proba(X)[:, 1]
        help_pts = m["help"].predict(X)
        final = m["final"].predict(X)
        modes = sorted(({"mode": md, "chance": float(p[k]), "help_points": float(help_pts[k]),
                         "final_function": float(final[k])} for k, md in enumerate(MODES)),
                       key=lambda d: (-d["chance"], -d["final_function"]))
        helpers = None
        by_sev = self.b["helpers"].get((area, disorder)) or self.b["helpers"].get((area, "stroke"))
        if by_sev:
            helpers = by_sev[min(by_sev, key=lambda s: abs(s - severity))]
        return {"area": area, "severity": severity, "disorder": disorder, "best": modes[0], "modes": modes,
                "helpers": helpers, "why": self.why(area), "metrics": self.b["metrics"]}

    def why(self, area: str, top: int = 4) -> list[str]:
        """Plain-language reasons: the most important area features, and where this area stands."""
        out = []
        cols = self.f.columns
        for name, _ in [kv for kv in self.b["importance"] if kv[0] in cols][:top]:
            j = cols.index(name)
            col = self.f.table[:, j]
            v = col[self.index[area]]
            if np.isnan(v):
                continue
            text = self.f.plain.get(name, name)
            if set(np.unique(col[~np.isnan(col)])) <= {0.0, 1.0}:      # a yes/no property
                out.append(f"{text}: {'yes' if v else 'no'}")
                continue
            pct = float((col[~np.isnan(col)] < v).mean() * 100)
            level = "high" if pct >= 67 else "low" if pct <= 33 else "middle"
            out.append(f"{text}: {level} (higher than {pct:.0f}% of areas)")
        return out
