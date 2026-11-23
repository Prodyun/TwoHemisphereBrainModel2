"""A self-contained 3D viewer file for one patient (no server needed; share or open offline).

    python scripts/patient_viewer.py --patient configs/patients/left_6d_stroke.yaml
    python scripts/patient_viewer.py --area L_6d --severity 0.9 --disorder stroke

Shows the AI's prediction and help pathways, and plays 4 months of care for the AI's plan,
standard care and the split brain (control): practice sessions, BCI detections, stimulation,
repetition counters, damage healing and the weekly recovery chart.
Output: results/plans/<name>/viewer.html
"""
import argparse
import sys
import webbrowser
from pathlib import Path

from brain_model.care_service import CareService
from brain_model.utils import load_config
from brain_model.visualization.care_view import care_page


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--patient", help="a patient file, e.g. configs/patients/left_6d_stroke.yaml")
    g.add_argument("--area", help="a damaged area, e.g. L_6d")
    ap.add_argument("--severity", type=float, default=0.9)
    ap.add_argument("--disorder", default="stroke")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    service = CareService(load_config("configs/hcp.yaml"), load_config("configs/care.yaml"))
    if args.patient:
        stem = Path(args.patient).stem
        service_preset = stem if Path(args.patient).parent.name == "patients" else None
        if service_preset is None:
            raise SystemExit("put the patient file in configs/patients/")
        payload = service.simulate(preset=stem)
        cfg, _ = service.config_for(preset=stem)
    else:
        stem = f"{args.area}_{args.disorder}_{args.severity:.2f}"
        payload = service.simulate(area=args.area, severity=args.severity, disorder=args.disorder)
        cfg, _ = service.config_for(args.area, args.severity, args.disorder)
    out = Path("results/plans") / stem / "viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(care_page(cfg, payload, interactive=False), encoding="utf-8")
    for c in payload["conditions"].values():
        m = c["measures"]
        print(f"{c['label']}: weeks to goal {m['weeks_to_goal']}, physical reps {m['physical_reps_to_goal']}, "
              f"final {m['final_function']:.0%}")
    print(f"Saved {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
