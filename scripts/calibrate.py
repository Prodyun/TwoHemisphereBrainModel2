"""Fit the model's guessed settings to PUBLISHED FORMULAS (docs/calibration.md).

    python scripts/calibrate.py                 # everything (~30-40 min on 10 cores)
    python scripts/calibrate.py --stage healing # floor_fraction, spontaneous_per_day, use_rate
    python scripts/calibrate.py --stage bridge  # plasticity_rate (formula), the brake (sweep + checks)
    python scripts/calibrate.py --quick         # small cohort, small grids (a check)

HEALING  van der Vliet 2020 (Ann Neurol): each virtual patient's share of possible recovery should follow
         sum_k P(k | FM0) r_k (1 - exp(-t / tau_k)), with the paper's 5 subgroups. CPASS (Dromerick 2021):
         20 extra practice hours at 2-3 months remove 15.8% of the starting deficit. A grid search over
         (floor_fraction, spontaneous_per_day, use_rate, floor_use_rate) minimises the misfit to both.
BRIDGE   plasticity_rate from Kokinovic & Medini 2018 (callosal takeover complete by day 14), via
         rate = 1 - 0.05^(1 / (14 x 24)). The brake (interhemispheric inhibition onto movement areas) is
         swept and checked against: NICHE / E-FIT (calming the healthy M1 is not better than sham on
         average), Di Pino 2014 (calming helps some patients with high reserve, fewer with low reserve)
         and the DCM sign pattern (Rehme 2011: the healthy M1's influence on the damaged M1 is supportive
         early and inhibitory late in poorly recovered patients). If Murase 2004 IHI numbers are filled in
         (configs/calibration/targets.yaml), k = (1/IHI - 1)/a is reported as well.

Uses only published summary numbers (configs/calibration/targets.yaml). Writes results/calibration/ only.
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from brain_model.brain import steps_per_day  # noqa: E402
from brain_model.calibration import expected_recovery, ihi_to_brake, rate_for_days  # noqa: E402
from brain_model.healthy_side import better, meaningful  # noqa: E402
from brain_model.patient import patient_config, patient_from_area  # noqa: E402
from brain_model.planning import healthy_for, standard_target, variant, with_targets  # noqa: E402
from brain_model.simulation.outcome import measures  # noqa: E402
from brain_model.simulation.runner import run_simulation  # noqa: E402
from brain_model.utils import load_config  # noqa: E402

AREAS = ["4", "6d", "3b", "6a", "1", "6mp"]
# (severity of the clicked area, neighbours, neighbours' share): from small to large strokes
SIZES = [(0.6, 4, 0.6), (0.9, 4, 0.6), (0.95, 10, 0.9)]


def cohort(base, care, quick=False):
    out = []
    for side in ("L_", "R_"):
        for a in (AREAS[:2] if quick else AREAS):
            for sev, nb, share in SIZES:
                p = patient_from_area(side + a, sev, "stroke", neighbors=nb, neighbor_share=share)
                p["name"] = f"{side}{a} s{sev} n{nb}"
                cfg = patient_config(base, care, p)
                cfg["therapy"]["function"] = "movement"
                out.append(cfg)
    return out


# ---------------------------------------------------------------- workers (run in other processes)

def _weekly(job):
    key, cfg = job
    run = run_simulation(cfg, with_lesion=True)
    m = measures(run, healthy_for(cfg), cfg)
    return key, m["weekly_function"], m["final_function"]


def _coupling(run, cfg, weeks):
    """DCM-style effective coupling from the healthy to the damaged movement areas: regress the damaged
    side's change dx(t) on [x_damaged(t), x_healthy_partner(t)] over the chosen weeks (non-practice steps)."""
    side = cfg["lesion"]["hemisphere"]
    other = "right" if side == "left" else "left"
    reg = run["functions"][side]["movement"]
    reg_h = run["functions"][other]["movement"]
    spd, onset = steps_per_day(cfg), cfg["lesion"]["start_step"]
    a, b = onset + int(weeks[0] * 7 * spd), onset + int(weeks[1] * 7 * spd)
    xd = run[side][a:b, reg].mean(axis=1)
    xh = run[other][a:b, reg_h].mean(axis=1)
    dx = np.diff(xd)
    X = np.column_stack([xd[:-1], xh[:-1], np.ones(len(dx))])
    coef, *_ = np.linalg.lstsq(X, dx, rcond=None)
    return float(coef[1])


def _bridge_job(job):
    key, cfg = job
    run = run_simulation(cfg, with_lesion=True)
    m = measures(run, healthy_for(cfg), cfg)
    return key, {"final_function": m["final_function"], "weeks_to_goal": m["weeks_to_goal"],
                 "weekly_function": m["weekly_function"],
                 "early": _coupling(run, cfg, (0, 2)), "late": _coupling(run, cfg, (12, 16))}


def pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with Pool(workers) as p:
        return list(p.imap_unordered(fn, jobs, chunksize=2))


# ---------------------------------------------------------------- HEALING

HEAL_KEYS = ("floor_fraction", "spontaneous_per_day", "use_rate", "floor_use_rate")


def heal_cfg(cfg, F, S, U, R=0.0, extra=None):
    c = variant(cfg, recovery={"floor_fraction": F, "spontaneous_per_day": S, "use_rate": U, "floor_use_rate": R})
    if extra:
        c = variant(c, therapy={"extra": extra})
    return c


def fit_healing(pts, targets, workers, quick, no_fine=False):
    groups = targets["healing"]["van_der_vliet_2020"]["groups"]
    cp = targets["healing"]["cpass_2021"]
    extra = {"start_day": cp["start_day"], "end_day": cp["start_day"] + cp["days"],
             "session_steps": max(1, round(cp["extra_hours"] / (cp["days"] * 5 / 7)))}
    # 1. each patient's starting function: no healing, no therapy (stable average over all weeks)
    base_jobs = [(i, variant(c, therapy={"mode": "none"}, recovery={"enabled": False})) for i, c in enumerate(pts)]
    f0 = {k: float(np.mean(w)) for k, w, _ in pool_map(_weekly, base_jobs, workers)}
    print("starting function (x66 = FM-UE):", {k: round(66 * v, 1) for k, v in sorted(f0.items())}, flush=True)
    # 2. the SAME patient and practice timetable with healing switched off, week by week. The weekly score
    #    depends on which hours of the job's rhythm fall outside practice; every patient shares the timetable,
    #    so those week-to-week swings are the same in both runs and cancel in (healed - unhealed).
    ref_jobs = [((i, "usual"), variant(c, recovery={"enabled": False})) for i, c in enumerate(pts)]
    ref_jobs += [((i, "extra"), variant(c, recovery={"enabled": False}, therapy={"extra": extra})) for i, c in enumerate(pts)]
    ref = {k: np.asarray(w) for k, w, _ in pool_map(_weekly, ref_jobs, workers)}
    severe = sorted(range(len(pts)), key=lambda i: f0[i])[: max(1, len(pts) // 3)]   # CPASS enrolled severe arms

    def score(grid):
        jobs = []
        for g in grid:
            for i, c in enumerate(pts):
                jobs.append(((g, i, "usual"), heal_cfg(c, *g)))
                if i in severe:
                    jobs.append(((g, i, "extra"), heal_cfg(c, *g, extra=extra)))
        res = {k: (w, fin) for k, w, fin in pool_map(_weekly, jobs, workers)}
        out = []
        for g in grid:
            errs, gains, curves = [], [], []
            for i in range(len(pts)):
                w = np.asarray(res[(g, i, "usual")][0])
                pot = max(1.0 - f0[i], 0.05)
                share = (w - ref[(i, "usual")]) / pot          # recovered share, timetable swings removed
                t = np.arange(len(w)) + 0.5
                tgt = expected_recovery(66 * f0[i], t, groups)
                errs.append(float(np.mean((share - tgt) ** 2)))
                curves.append((share, tgt))
                if i in severe:
                    we = np.asarray(res[(g, i, "extra")][0])
                    end = min(int((cp["start_day"] + cp["days"]) / 7) + 2, len(w) - 1)
                    gain_e = np.mean(we[end - 1:end + 1] - ref[(i, "extra")][end - 1:end + 1])
                    gain_u = np.mean(w[end - 1:end + 1] - ref[(i, "usual")][end - 1:end + 1])
                    gains.append(float((gain_e - gain_u) / pot))
            vdv = float(np.mean(errs))
            cpass = float(np.mean(gains))
            loss = vdv + (cpass - cp["gain_share_of_deficit"]) ** 2
            out.append({"floor_fraction": g[0], "spontaneous_per_day": g[1], "use_rate": g[2], "floor_use_rate": g[3],
                        "vdv_mse": round(vdv, 5), "cpass_gain_share": round(cpass, 4), "loss": round(loss, 5),
                        "mean_share": np.mean([c[0] for c in curves], axis=0).round(3).tolist(),
                        "mean_target": np.mean([c[1] for c in curves], axis=0).round(3).tolist()})
        return sorted(out, key=lambda r: r["loss"])

    # floor_use_rate (Rs): practice lowers the permanent floor; it is what lets extra practice at 2-3 months
    # (CPASS) raise the final level after usual care has levelled off. 0 = the old rule.
    if quick:
        Fs, Ss, Us, Rs = [0.2], [0.1], [0.3], [0.0, 0.01, 0.1]
    else:
        Fs, Ss, Us, Rs = [0.2, 0.3, 0.45], [0.03, 0.1], [0.03, 0.3], [0.0, 0.003, 0.01, 0.03, 0.1]
    t0 = time.time()
    coarse = score(list(itertools.product(Fs, Ss, Us, Rs)))
    print(f"coarse grid done in {time.time() - t0:.0f}s; best: {coarse[0]}", flush=True)
    b = coarse[0]
    r0 = b['floor_use_rate'] or 0.001
    fine_grid = sorted({(b['floor_fraction'], round(b['spontaneous_per_day'] * mS, 5), b['use_rate'], round(r0 * mR, 5))
                        for mS in (0.6, 1.0, 1.6) for mR in (0.5, 1.0, 2.0)})
    fine = score(fine_grid) if not (quick or no_fine) else coarse
    best = fine[0]
    print("best:", best, flush=True)
    return {"f0": f0, "coarse": coarse[:10], "fine": fine[:10], "best": best, "extra": extra}


# ---------------------------------------------------------------- BRIDGE

def fit_bridge(base, care, targets, workers, quick, healing=None):
    spd = steps_per_day({"time": care["time"]})
    days = targets["bridge"]["kokinovic_medini_2018"]["days_to_takeover"]
    rate = rate_for_days(days, spd)
    print(f"plasticity_rate from Kokinovic & Medini (95% by day {days}): {rate:.5f}", flush=True)
    care2 = json.loads(json.dumps(care))
    care2.setdefault("corpus_callosum", {})["plasticity_rate"] = round(rate, 5)
    if healing:
        care2["recovery"].update({k: healing[k] for k in HEAL_KEYS})
    pts = []
    for side in ("L_", "R_"):
        for a in (AREAS[:2] if quick else AREAS):
            for sev in (0.6, 0.9):
                p = patient_from_area(side + a, sev, "stroke")
                p["name"] = f"{side}{a} s{sev}"
                c = patient_config(base, care2, p)
                c["therapy"]["function"] = "movement"
                pts.append(c)
    ks = [0.0, 1.0, 4.0] if quick else [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    jobs = []
    for k in ks:
        for i, c in enumerate(pts):
            ck = variant(c, corpus_callosum={"inhibition": k, "inhibition_jobs": ["movement"]})
            std = standard_target(ck, care)
            jobs += [((k, i, "practice"), ck),
                     ((k, i, "calm"), with_targets(ck, [{"area": std["area"], "strength": -0.3}])),
                     ((k, i, "excite"), with_targets(ck, [{"area": std["area"], "strength": 0.5}]))]
    res = dict(pool_map(_bridge_job, jobs, workers))
    tol = meaningful(pts[0])
    sweep = []
    for k in ks:
        calm_diff, words, late_poor, early_all, finals = [], {}, [], [], []
        for i, c in enumerate(pts):
            pr, ca, ex = res[(k, i, "practice")], res[(k, i, "calm")], res[(k, i, "excite")]
            calm_diff.append(ca["final_function"] - pr["final_function"])
            best, word = pr, "LEAVE"
            for kind, row in (("EXCITE", ex), ("CALM", ca)):
                if better(row, best, tol):
                    best, word = row, kind
            sev = c["patient"]["name"].split(" s")[-1]
            words.setdefault(sev, []).append(word)
            early_all.append(pr["early"])
            finals.append(pr["final_function"])
        order = np.argsort(finals)
        poor = order[: max(1, len(order) // 3)]          # the worst-recovered third
        late_poor = [res[(k, int(i), "practice")]["late"] for i in poor]
        calm = {s: w.count("CALM") / len(w) for s, w in words.items()}
        row = {"inhibition": k,
               "calm_vs_sham_points": round(100 * float(np.mean(calm_diff)), 2),
               "calm_share_by_severity": {s: round(v, 2) for s, v in calm.items()},
               "dcm_early_mean": round(float(np.mean(early_all)), 4),
               "dcm_late_poor_mean": round(float(np.mean(late_poor)), 4),
               "mean_final_function": round(float(np.mean(finals)), 4)}
        row["niche_ok"] = abs(row["calm_vs_sham_points"]) < 2.0
        row["dipino_ok"] = calm.get("0.6", 0) > 0 and calm.get("0.6", 0) >= calm.get("0.9", 0)
        row["dcm_ok"] = row["dcm_early_mean"] > 0 and row["dcm_late_poor_mean"] < row["dcm_early_mean"]
        row["all_ok"] = row["niche_ok"] and row["dipino_ok"] and row["dcm_ok"]
        sweep.append(row)
        print(row, flush=True)
    ok = [r for r in sweep if r["all_ok"]]
    chosen = min(ok, key=lambda r: r["inhibition"]) if ok else None
    m = targets["bridge"]["murase_2004"]
    murase_k = None
    if m.get("ihi_ratio_patients"):
        murase_k = ihi_to_brake(float(m["ihi_ratio_patients"]), 0.3)
    return {"plasticity_rate": round(rate, 5), "sweep": sweep,
            "chosen_inhibition": chosen["inhibition"] if chosen else 0.0, "brake_found": bool(chosen),
            "murase_k": murase_k}


# ---------------------------------------------------------------- report

def report(out, od: Path):
    L = ["# Calibration: guessed settings replaced by published formulas", "",
         "Simulation only; numbers from papers are in configs/calibration/targets.yaml.", ""]
    h = out.get("healing")
    if h:
        b = h["best"]
        L += ["## Healing (van der Vliet 2020 + CPASS 2021)", "",
              f"Best fit: floor_fraction **{b['floor_fraction']}**, spontaneous_per_day **{b['spontaneous_per_day']}**, "
              f"use_rate **{b['use_rate']}**, floor_use_rate **{b['floor_use_rate']}**.", "",
              f"- Misfit to the van der Vliet recovery curves (mean squared error of the recovered share): {b['vdv_mse']}",
              f"- CPASS: extra practice removed {100 * b['cpass_gain_share']:.1f}% of the starting deficit "
              f"in the most severe third (trial: 15.8%)", "",
              "| week | model share of possible recovery | van der Vliet expected |", "|---|---|---|"]
        for w, (a, t) in enumerate(zip(b["mean_share"], b["mean_target"])):
            L.append(f"| {w + 1} | {a:.2f} | {t:.2f} |")
        L += ["", "Top 5 settings tried:", "", "| floor | spontaneous/day | use_rate | floor_use_rate | vdV MSE | CPASS share | loss |",
              "|---|---|---|---|---|---|---|"]
        for r in h["fine"][:5]:
            L.append(f"| {r['floor_fraction']} | {r['spontaneous_per_day']} | {r['use_rate']} | {r['floor_use_rate']} | {r['vdv_mse']} | "
                     f"{r['cpass_gain_share']} | {r['loss']} |")
        L.append("")
    br = out.get("bridge")
    if br:
        L += ["## Bridge (Kokinovic & Medini 2018; Murase 2004; NICHE/E-FIT; Di Pino 2014; Rehme 2011)", "",
              f"plasticity_rate = 1 - 0.05^(1/(14 x 24)) = **{br['plasticity_rate']}** (callosal takeover 95% done by day 14).", "",
              "| brake k | calm vs sham (points) | CALM chosen (sev 0.6 / 0.9) | DCM early | DCM late (poor) | NICHE | Di Pino | DCM |",
              "|---|---|---|---|---|---|---|---|"]
        for r in br["sweep"]:
            cs = r["calm_share_by_severity"]
            yn = lambda b: "✔" if b else "✖"   # noqa: E731
            L.append(f"| {r['inhibition']} | {r['calm_vs_sham_points']:+.2f} | {cs.get('0.6', 0):.0%} / {cs.get('0.9', 0):.0%} | "
                     f"{r['dcm_early_mean']:+.4f} | {r['dcm_late_poor_mean']:+.4f} | {yn(r['niche_ok'])} | "
                     f"{yn(r['dipino_ok'])} | {yn(r['dcm_ok'])} |")
        L += ["", (f"Chosen brake: **{br['chosen_inhibition']}** (smallest value passing all three checks)."
                   if br["brake_found"] else "No brake value passed all three checks: the brake stays OFF (reported honestly)."),
              ""]
        if br.get("murase_k") is not None:
            L.append(f"Murase 2004 IHI ratio gives k = (1/IHI - 1)/a = {br['murase_k']:.2f} (a = 0.3).")
    (od / "report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["healing", "bridge", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-fine", action="store_true", help="skip the fine grid around the best coarse point")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="results/calibration")
    args = ap.parse_args()
    os.chdir(ROOT)
    base, care = load_config("configs/hcp.yaml"), load_config("configs/care.yaml")
    targets = yaml.safe_load(open("configs/calibration/targets.yaml", encoding="utf-8"))
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    out = {}
    if args.stage in ("healing", "all"):
        pts = cohort(base, care, args.quick)
        print(f"{len(pts)} virtual patients", flush=True)
        out["healing"] = fit_healing(pts, targets, args.workers, args.quick, args.no_fine)
    if args.stage in ("bridge", "all"):
        out["bridge"] = fit_bridge(base, care, targets, args.workers, args.quick,
                                   out.get("healing", {}).get("best"))
    fitted = {}
    if "healing" in out:
        fitted["recovery"] = {k: out["healing"]["best"][k] for k in HEAL_KEYS}
    if "bridge" in out:
        fitted["corpus_callosum"] = {"plasticity_rate": out["bridge"]["plasticity_rate"],
                                     "inhibition": out["bridge"]["chosen_inhibition"],
                                     "inhibition_jobs": ["movement"]}
    (od / "fitted.yaml").write_text(yaml.safe_dump(fitted, sort_keys=False), encoding="utf-8")
    (od / "results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    report(out, od)
    print(json.dumps(fitted, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
