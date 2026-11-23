"""The logic behind the interactive explorer's patient-care mode (also usable without a browser).

  predict(area, severity, disorder)   the AI Help Predictor's answer: chance of help per practice
                                      mode, expected help, and which healthy-side areas carry it
                                      (falls back to tracing a quick simulation if the model is
                                      not trained yet)
  plans(...)                          the plans to compare: the AI's recommended plan, standard care
                                      (physical practice, no stimulation), the recommended plan in a
                                      split brain (control) and, optionally, the judge's own plan
  simulate(...)                       runs the plans and returns viewer data (care_view.care_payload)
"""
from pathlib import Path

from .help_data import MODES as DATA_MODES
from .help_model import MODEL_FILE, HelpPredictor
from .help_pathways import trace
from .patient import load_patient, patient_config, patient_from_area
from .planning import healthy_for, variant, with_targets
from .simulation import run_simulation
from .visualization.care_view import care_payload

DISORDERS = ("stroke", "tumor", "spreading")
PRESETS = {
    "left_6d_stroke": "Motor stroke (left 6d, right hand)",
    "left_broca_stroke": "Language stroke (left Broca)",
    "right_ffc_tumor": "Tumour (right face area)",
}


class CareService:
    def __init__(self, base: dict, care: dict, model_file: Path = MODEL_FILE):
        self.base, self.care = base, care
        self.predictor = HelpPredictor.load(model_file) if Path(model_file).exists() else None

    # ---------- a patient ----------
    def config_for(self, area: str | None = None, severity: float = 0.9, disorder: str = "stroke",
                   preset: str | None = None) -> tuple[dict, dict]:
        if disorder not in DISORDERS:
            raise ValueError(f"disorder must be one of {DISORDERS}")
        if not 0.1 <= float(severity) <= 0.98:
            raise ValueError("severity must be between 0.1 and 0.98")
        if preset:
            if preset not in PRESETS:
                raise ValueError(f"unknown preset {preset}")
            patient = load_patient(Path("configs/patients") / f"{preset}.yaml")
        else:
            # a spreading disease starts in the clicked area only; a stroke or tumour also hits neighbours
            patient = patient_from_area(area, float(severity), disorder, neighbors=0 if disorder == "spreading" else 4)
        return patient_config(self.base, self.care, patient), patient

    # ---------- the AI's answer ----------
    def predict(self, area: str, severity: float = 0.9, disorder: str = "stroke") -> dict:
        if self.predictor is not None:
            out = self.predictor.predict(area, float(severity), "stroke" if disorder == "spreading" else disorder)
            out["disorder"] = disorder
            out["source"] = ("AI model (trained on simulated patients, tested on unseen areas)" if disorder != "spreading"
                             else "AI model trained on stroke and tumour: for a spreading disease this is an estimate")
            if out.get("helpers") is None:
                out["helpers"] = self.trace_helpers(area, severity, disorder)
            return out
        return {"area": area, "severity": severity, "disorder": disorder, "best": None, "modes": [],
                "helpers": self.trace_helpers(area, severity, disorder), "why": [],
                "source": "pathway tracer only (the AI model is not trained yet)"}

    def trace_helpers(self, area: str, severity: float, disorder: str) -> list:
        cfg, _ = self.config_for(area, severity, disorder)
        run = run_simulation(cfg, with_lesion=True)
        return trace(cfg, run)["helpers"]

    # ---------- the plans to compare ----------
    def plans(self, cfg: dict, prediction: dict | None, judge: dict | None = None) -> list[tuple[str, str, dict]]:
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
                 prediction: dict | None = None) -> dict:
        cfg, patient = self.config_for(area, severity, disorder, preset)
        if start_day is not None:
            cfg = variant(cfg, therapy={"start_day": int(start_day)}, stimulation={"start_day": int(start_day)})
        side = cfg["lesion"]["hemisphere"]
        on_side = {k: v for k, v in cfg["lesion"]["damage_map"].items() if k.startswith("L_" if side == "left" else "R_")}
        epicentre = max(on_side, key=on_side.get)
        if prediction is None:
            prediction = self.predict(epicentre, max(on_side.values()), patient.get("disorder", disorder))
        plans = self.plans(cfg, prediction, judge)
        payload = care_payload(cfg, plans, prediction, prediction.get("helpers"),
                               {"name": patient.get("name"), "epicentre": epicentre, "disorder": patient.get("disorder")})
        return payload

    def warm(self, cfg: dict) -> None:
        healthy_for(cfg)
