"""The logic behind the interactive explorer's Clinic (also usable without a browser).

  diseases()                          the disease tabs (configs/diseases/*.yaml)
  predict(area, severity, disorder)   the AI Help Predictor's answer (stroke / tumour, where it was trained):
                                      chance of help per practice mode, expected help, helper areas.
                                      Other diseases: the pathway tracer only ("simulation only")
  plans(...)                          the plans to compare: AI plan, standard care, split brain (+ judge)
                                      for AI diseases; the profile's own treatments otherwise
  simulate(...)                       runs the plans and returns viewer data (care_view.care_payload)
  intake(...)                         an uploaded patient (mask / table / symptoms) -> damage map + card
  risk(...)                           the spread-risk map (spread_risk.py)
  options(...)                        the options race: every therapy ranked, with a dose curve and badges
  healthy_side(...)                   Bridge Builder, Mirror Drive, Help-or-Hinder, Hand-off
"""
import copy
from pathlib import Path

from . import diseases as registry
from .help_data import MODES as DATA_MODES
from .help_model import MODEL_FILE, HelpPredictor
from .help_pathways import trace
from .patient import deep_merge, load_patient, patient_config, patient_from_area, problem_card
from .planning import healthy_for, rank_key, run_plans, standard_target, variant, with_targets
from .simulation import run_simulation
from .utils import load_config
from .visualization.care_view import care_payload

PRESETS = {
    "left_6d_stroke": "Motor stroke (left 6d, right hand)",
    "left_broca_stroke": "Language stroke (left Broca)",
    "right_ffc_tumor": "Tumour (right face area)",
}
EVIDENCE_FILE = Path("configs/evidence.yaml")


def set_path(cfg: dict, key: str, value) -> dict:
    """Set e.g. 'lesion.cross_spread' = 0.5 in a copy of cfg."""
    out = copy.deepcopy(cfg)
    node = out
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


class CareService:
    def __init__(self, base: dict, care: dict, model_file: Path = MODEL_FILE):
        self.base, self.care = base, care
        self.predictor = HelpPredictor.load(model_file) if Path(model_file).exists() else None
        self.evidence = load_config(EVIDENCE_FILE)["options"] if EVIDENCE_FILE.exists() else {}

    # ---------- the disease tabs ----------
    def diseases(self) -> list[dict]:
        return registry.tabs()

    def profile(self, disorder: str) -> dict:
        p = registry.get(disorder)
        if p is None:
            raise ValueError(f"unknown disease '{disorder}' (choose from {sorted(registry.profiles())})")
        return p

    # ---------- a patient ----------
    def config_for(self, area: str | None = None, severity: float = 0.9, disorder: str = "stroke",
                   preset: str | None = None, overrides: dict | None = None,
                   patient: dict | None = None) -> tuple[dict, dict]:
        if not 0.05 <= float(severity) <= 0.98:
            raise ValueError("severity must be between 0.05 and 0.98")
        if patient is not None:
            patient = {**patient, "disorder": patient.get("disorder", disorder)}
            self.profile(patient["disorder"])
        elif preset:
            if preset not in PRESETS:
                raise ValueError(f"unknown preset {preset}")
            patient = load_patient(Path("configs/patients") / f"{preset}.yaml")
        else:
            profile = self.profile(disorder)
            if disorder in ("stroke", "tumor", "spreading"):
                # a spreading disease starts in the clicked area only; a stroke or tumour also hits neighbours
                patient = patient_from_area(area or profile.get("default_area"), float(severity), disorder,
                                            neighbors=0 if disorder == "spreading" else 4)
            else:
                patient = registry.make_patient(profile, area, float(severity))
        cfg = patient_config(self.base, self.care, patient)
        for key, value in (overrides or {}).items():
            if key != "severity":
                cfg = set_path(cfg, key, value)
        return cfg, patient

    # ---------- the AI's answer ----------
    def predict(self, area: str, severity: float = 0.9, disorder: str = "stroke") -> dict:
        profile = registry.get(disorder) or {}
        trained = bool(profile.get("ai")) or disorder == "spreading"
        if self.predictor is not None and trained:
            out = self.predictor.predict(area, float(severity), "stroke" if disorder == "spreading" else disorder)
            out["disorder"] = disorder
            out["source"] = ("AI model (trained on simulated patients, tested on unseen areas)" if disorder != "spreading"
                             else "AI model trained on stroke and tumour: for a spreading disease this is an estimate")
            if out.get("helpers") is None:
                out["helpers"] = self.trace_helpers(area, severity, disorder)
            return out
        return {"area": area, "severity": severity, "disorder": disorder, "best": None, "modes": [],
                "helpers": self.trace_helpers(area, severity, disorder), "why": [],
                "source": ("simulation only: the AI is not trained on this disease" if self.predictor is not None
                           else "pathway tracer only (the AI model is not trained yet)")}

    def trace_helpers(self, area: str, severity: float, disorder: str, cfg: dict | None = None) -> list:
        if cfg is None:
            cfg, _ = self.config_for(area, severity, disorder)
        run = run_simulation(cfg, with_lesion=True)
        return trace(cfg, run)["helpers"]

    # ---------- the plans to compare ----------
    def plans(self, cfg: dict, prediction: dict | None, judge: dict | None = None) -> list[tuple[str, str, dict]]:
        disorder = cfg["patient"]["disorder"]
        profile = registry.get(disorder) or {}
        if not profile.get("ai") and disorder != "spreading":
            out = registry.plan_list(cfg, profile)
        else:
            best_mode = (prediction or {}).get("best") or {"mode": "bci+vns"}
            mode = DATA_MODES.get(best_mode["mode"], best_mode["mode"])
            helpers = (prediction or {}).get("helpers") or []
            targets = [{"area": h["area"], "strength": 0.5} for h in helpers[:2]]
            rec = with_targets(variant(cfg, therapy={"mode": mode}), targets)
            label = f"AI plan: {best_mode['mode']}" + (f" + excite {', '.join(t['area'] for t in targets)}" if targets else "")
            out = [("recommended", label, rec),
                   ("standard_care", "Standard care: physical practice", variant(cfg, therapy={"mode": "physical"})),
                   ("split", "AI plan, split brain (control)", variant(rec, corpus_callosum={"strength": 0.0}))]
        if judge:
            jmode = DATA_MODES.get(judge.get("mode", "physical"), judge.get("mode", "physical"))
            jt = [{"area": t["area"], "strength": float(t.get("strength", 0.5))} for t in judge.get("targets", [])]
            out.insert(1, ("judge", "Judge's plan: " + judge.get("mode", "physical") +
                           (f" + {', '.join(t['area'] for t in jt)}" if jt else ""),
                           with_targets(variant(cfg, therapy={"mode": jmode}), jt)))
        return out

    def simulate(self, area: str | None = None, severity: float = 0.9, disorder: str = "stroke",
                 preset: str | None = None, start_day: int | None = None, judge: dict | None = None,
                 prediction: dict | None = None, overrides: dict | None = None,
                 patient: dict | None = None) -> dict:
        cfg, patient = self.config_for(area, severity, disorder, preset, overrides, patient)
        if start_day is not None:
            cfg = variant(cfg, therapy={"start_day": int(start_day)}, stimulation={"start_day": int(start_day)})
        side = cfg["lesion"]["hemisphere"]
        on_side = {k: v for k, v in cfg["lesion"]["damage_map"].items() if k.startswith("L_" if side == "left" else "R_")}
        epicentre = max(on_side, key=on_side.get)
        name = patient.get("disorder", disorder)
        profile = registry.get(name)
        if prediction is None:
            prediction = self.predict(epicentre, max(on_side.values()), name) if (profile or {}).get("ai") \
                else {"area": epicentre, "disorder": name, "best": None, "modes": [], "why": [],
                      "helpers": self.trace_helpers(epicentre, max(on_side.values()), name, cfg),
                      "source": "simulation only: the AI is not trained on this disease"}
        plans = self.plans(cfg, prediction, judge)
        payload = care_payload(cfg, plans, prediction, prediction.get("helpers"),
                               {"name": patient.get("name"), "epicentre": epicentre, "disorder": name,
                                "notes": patient.get("notes", []), "source": patient.get("source")},
                               profile=profile)
        payload["meta"]["care"]["card"] = problem_card(cfg)
        payload["meta"]["care"]["disease"] = next((t for t in registry.tabs() if t["name"] == name), {"name": name})
        return payload

    # ---------- uploaded patient ----------
    def intake(self, kind: str, data: dict) -> dict:
        from . import intake as ik
        disorder = data.get("disorder", "stroke")
        self.profile(disorder)
        if kind == "mask":
            patient = ik.from_mask(data["path"], disorder)
        elif kind == "table":
            patient = ik.from_table(ik.read_table(data["text"]), data.get("value_kind", "damage"), disorder)
        elif kind == "symptoms":
            patient = ik.from_symptoms(data.get("symptoms", []), self.base, float(data.get("severity", 0.7)),
                                       data.get("eeg"), disorder)
        else:
            raise ValueError("kind must be 'mask', 'table' or 'symptoms'")
        cfg = patient_config(self.base, self.care, patient)
        return {"patient": patient, "card": problem_card(cfg)}

    # ---------- spread risk ----------
    def risk(self, n_runs: int = 30, **kw) -> dict:
        from .spread_risk import risk
        cfg, patient = self.config_for(**kw)
        return risk(cfg, registry.get(patient.get("disorder", "stroke")), n_runs)

    # ---------- the options race ----------
    def _ev(self, key: str) -> dict:
        e = self.evidence.get(key, {})
        return {"key": key, "name": e.get("name", key), "status": e.get("status", "model"),
                "basis": e.get("basis", ""), "source": e.get("source", "")}

    def options(self, workers: int = 4, dose: tuple = (0.2, 0.35, 0.5, 0.75, 1.0), **kw) -> dict:
        """Every therapy for this patient, ranked by weeks to recover (then final function)."""
        from .healthy_side import bridge_pairs, mirror_pairs
        cfg, patient = self.config_for(**kw)
        healthy_for(cfg)
        run = run_simulation(cfg, with_lesion=True)
        helpers = trace(cfg, run)["helpers"]
        helper = helpers[0]["area"] if helpers else None
        std = standard_target(cfg, self.care)
        base = variant(cfg, therapy={"mode": "physical"})
        jobs = [("physical", "option", base),
                ("imagery", "option", variant(cfg, therapy={"mode": "imagery"})),
                ("bci_healthy", "option", variant(cfg, therapy={"mode": "bci", "bci": {"decoder": "healthy"}})),
                ("vns", "option", variant(cfg, therapy={"mode": ["physical", "vns"]})),
                ("calm_healthy", "option", with_targets(base, [{"area": std["area"], "strength": -0.3}]))]
        if helper:
            jobs.append(("excite_helper", "option", with_targets(base, [{"area": helper, "strength": 0.5}])))
        bp = bridge_pairs(cfg, helpers, k=1)
        if bp:
            jobs.append(("bridge_builder", "option",
                         variant(base, bridge={**self.care.get("bridge", {}), "mode": "pair", "pairs": bp})))
        mp = mirror_pairs(cfg)
        if mp:
            jobs.append(("mirror_drive", "option",
                         variant(base, bridge={**self.care.get("bridge", {}), "mode": "mirror_drive", "pairs": mp})))
        jobs.append(("split", "control", variant(base, corpus_callosum={"strength": 0.0})))
        if helper:
            for s in dose:
                jobs.append((f"dose {s}", "dose", with_targets(base, [{"area": helper, "strength": s}])))
        rows = run_plans(jobs, workers)
        opts = []
        for r in rows:
            if r["kind"] == "dose":
                continue
            e = self._ev(r["plan"])
            detail = {"excite_helper": helper, "calm_healthy": std["area"],
                      "bridge_builder": f"{bp[0]['from']} → {bp[0]['to']}" if bp else None,
                      "mirror_drive": ", ".join(f"{p['from']} → {p['to']}" for p in mp) if mp else None}.get(r["plan"])
            opts.append({**e, "detail": detail, "weeks": r["weeks_to_goal"], "final": round(r["final_function"], 4),
                         "reps": r["physical_reps_to_goal"], "control": r["kind"] == "control"})
        ranked = sorted([o for o in opts if not o["control"]],
                        key=lambda o: rank_key({"weeks_to_goal": o["weeks"], "physical_reps_to_goal": 0.0,
                                                "final_function": o["final"]}))
        curve = [{"strength": float(r["plan"].split()[1]), "weeks": r["weeks_to_goal"],
                  "final": round(r["final_function"], 4)} for r in rows if r["kind"] == "dose"]
        knee = _knee(curve)
        return {"patient": patient.get("name"), "options": ranked + [o for o in opts if o["control"]],
                "dose": {"area": helper, "curve": curve, "suggested": knee}}

    # ---------- the healthy side helps ----------
    def healthy_side(self, workers: int = 4, **kw) -> dict:
        from . import healthy_side as hs
        cfg, patient = self.config_for(**kw)
        healthy_for(cfg)
        helpers = trace(cfg, run_simulation(cfg, with_lesion=True))["helpers"]
        bb = hs.bridge_builder(cfg, self.care, helpers, workers)
        md = hs.mirror_drive(cfg, self.care, workers)
        hh = hs.help_or_hinder(cfg, self.care, workers)
        target = hh["area"] if hh["word"] != "LEAVE" else (helpers[0]["area"] if helpers else hh["area"])
        sign = -0.3 if hh["word"] == "CALM" else 0.5
        ho = hs.hand_off(cfg, [{"area": target, "strength": sign}], workers)
        def slim(rows):
            return [{"plan": r["plan"], "kind": r["kind"], "weeks": r["weeks_to_goal"],
                     "final": round(r["final_function"], 4), **({"help_days": r["help_days"]} if "help_days" in r else {})}
                    for r in rows]
        none = bb["none"]
        best = bb["best"][0] if bb["best"] else None
        mdr = {r["kind"]: r for r in md["rows"]}
        return {
            "patient": patient.get("name"),
            "bridge": {"best": best["plan"] if best else None, "helps": bb["helps"],
                       "weeks": best["weeks_to_goal"] if best else None, "weeks_without": none["weeks_to_goal"],
                       "rows": slim(bb["rows"])},
            "mirror": {"helps": md["helps"], "beats_timing": md["beats_timing"], "duty": round(md["duty"], 3),
                       "weeks": mdr["closed"]["weeks_to_goal"], "weeks_without": mdr["none"]["weeks_to_goal"],
                       "pairs": [f"{p['from']} → {p['to']}" for p in md["pairs"]], "rows": slim(md["rows"])},
            "switch": {"word": hh["word"], "area": hh["area"], "why": hh["why"], "reserve": hh["reserve"],
                       "rows": slim(hh["rows"])},
            "handoff": {"best": ho["best"]["plan"] if ho["best"] else None, "helps": ho["helps"],
                        "target": target, "rows": slim(ho["rows"])},
        }


def _knee(curve: list[dict]) -> float | None:
    """The smallest dose that gets (almost) the best result: within 1 point of the best final function and
    no slower than the fastest."""
    if not curve:
        return None
    fastest = min((c["weeks"] for c in curve if c["weeks"] is not None), default=None)
    best = max(c["final"] for c in curve)
    for c in sorted(curve, key=lambda c: c["strength"]):
        if c["final"] >= best - 0.01 and (fastest is None or c["weeks"] == fastest):
            return c["strength"]
    return None
