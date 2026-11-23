"""Explore the model: click any of the 360 areas (either hemisphere).

    python scripts/explore.py                      # patient-care mode with the AI (default)
    python scripts/explore.py --booth              # large fonts for a science-fair booth
    python scripts/explore.py --disorder-mode      # the original "start a disorder here" mode
    python scripts/explore.py --config configs/hcp.yaml --port 8765

Patient-care mode: click an area -> the AI Help Predictor says how likely the other hemisphere can
help it, which areas carry the help, and which practice works best; "Simulate" then runs 4 months of
care for the AI's plan, standard care and the split brain (and the judge's own plan, if chosen).

Starts a small web server on this laptop only (127.0.0.1), opens the viewer in your browser, and
runs a new simulation (connected brain vs split brain, about 3 seconds) each time you press
"Start disorder here". No internet is needed except to load the 3D library the first time.
Keep this window open while you use the viewer; press Ctrl+C here to stop.
"""
import argparse
import copy
import json
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_model.care_service import CareService  # noqa: E402
from brain_model.hcp import load_hcp_brain  # noqa: E402
from brain_model.visualization.care_view import care_page  # noqa: E402
from brain_model.simulation import run_experiment, summarize  # noqa: E402
from brain_model.utils import load_config  # noqa: E402
from brain_model.visualization.viewer import run_payload, viewer_html, viewer_payload  # noqa: E402

LOCK = threading.Lock()  # one simulation at a time


def area_names(config: dict, side: str) -> list[str]:
    return load_hcp_brain(config["anatomy"]["file"]).side_names(side)


def simulate(config: dict) -> tuple[dict, dict]:
    results = run_experiment(config)
    return results, summarize(results, config)


def handle_run(base: dict, request: dict) -> tuple[int, dict]:
    """Run one experiment for a chosen start. Returns (HTTP status, JSON body)."""
    side = request.get("hemisphere")
    region = str(request.get("region", ""))
    kind = request.get("type", "spreading")
    if side not in ("left", "right"):
        return 400, {"error": "hemisphere must be 'left' or 'right'"}
    if kind not in ("spreading", "stroke"):
        return 400, {"error": "type must be 'spreading' or 'stroke'"}
    names = area_names(base, side)
    prefix = "L_" if side == "left" else "R_"
    name = region if region.startswith(prefix) else prefix + region
    if name not in names:
        return 400, {"error": f"unknown {side} area: {region}"}
    cfg = copy.deepcopy(base)
    cfg["lesion"].update(hemisphere=side, region=name, type=kind)
    t0 = time.time()
    with LOCK:
        results, summary = simulate(cfg)
    benefit = summary["benefit_of_callosum"] * 100
    print(f"{side} {name.removeprefix(prefix)} ({kind}): connected "
          f"{summary['connected']['damaged_side_score']:.0%} vs split {summary['split']['damaged_side_score']:.0%}"
          f" -> help from the other side {benefit:+.1f} points  [{time.time() - t0:.1f} s]", flush=True)
    return 200, run_payload(cfg, results, summary)


def handle_predict(service: CareService, request: dict) -> tuple[int, dict]:
    area = str(request.get("area", ""))
    if not area.startswith(("L_", "R_")):
        return 400, {"error": "area must be a full name like L_6d"}
    t0 = time.time()
    with LOCK:
        out = service.predict(area, float(request.get("severity", 0.9)), request.get("disorder", "stroke"))
    best = out.get("best") or {}
    print(f"predict {area}: best {best.get('mode')} {best.get('chance', 0):.0%}  [{time.time() - t0:.1f} s]", flush=True)
    return 200, out


def handle_simulate(service: CareService, request: dict) -> tuple[int, dict]:
    t0 = time.time()
    with LOCK:
        payload = service.simulate(area=request.get("area"), severity=float(request.get("severity", 0.9)),
                                   disorder=request.get("disorder", "stroke"), preset=request.get("preset") or None,
                                   start_day=request.get("start_day"), judge=request.get("judge"),
                                   prediction=request.get("prediction"))
    summary = ", ".join(f"{c['label'].split(':')[0]}: week {c['measures']['weeks_to_goal']}"
                        for c in payload["conditions"].values())
    print(f"simulate {payload['meta']['care']['patient'].get('epicentre')}: {summary}  [{time.time() - t0:.1f} s]",
          flush=True)
    return 200, payload


def make_handler(base: dict, first_page: str, service: CareService | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.split("#")[0] in ("/", "/index.html"):
                self._send(200, first_page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            routes = {"/api/run": lambda r: handle_run(base, r)}
            if service is not None:
                routes["/api/predict"] = lambda r: handle_predict(service, r)
                routes["/api/simulate"] = lambda r: handle_simulate(service, r)
            if self.path not in routes:
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                status, body = routes[self.path](request)
            except Exception as exc:  # report the problem in the viewer instead of crashing
                status, body = 500, {"error": f"{type(exc).__name__}: {exc}"}
            self._send(status, json.dumps(body, separators=(",", ":")).encode("utf-8"), "application/json")

        def log_message(self, *args):  # keep the console for results only
            pass

    return Handler


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "hcp.yaml"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--booth", action="store_true", help="large fonts for a science-fair booth")
    parser.add_argument("--disorder-mode", action="store_true", help="the original spreading-disorder explorer")
    parser.add_argument("--care", default=str(ROOT / "configs" / "care.yaml"))
    args = parser.parse_args()

    import os
    os.chdir(ROOT)  # config paths (e.g. data/processed/hcp_brain.npz) are relative to the project
    base = load_config(args.config)
    service = None
    if args.disorder_mode:
        print("Preparing the first simulation ...", flush=True)
        results, summary = simulate(base)
        page = viewer_html(viewer_payload(base, results, summary, interactive=True))
        how = "Click an area on either hemisphere, then 'Start disorder here'."
    else:
        service = CareService(base, load_config(args.care))
        print("AI model:", "loaded" if service.predictor else "not trained yet (pathway tracer only)", flush=True)
        print("Preparing the first patient (motor stroke, left 6d) ...", flush=True)
        payload = service.simulate(preset="left_6d_stroke")
        cfg, _ = service.config_for(preset="left_6d_stroke")
        page = care_page(cfg, payload, interactive=True)
        how = "Click any area: the AI predicts whether the other side can help; then press 'Simulate'."
    port = free_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(base, page, service))
    url = f"http://127.0.0.1:{port}/" + ("#booth" if args.booth else "")
    print(f"Viewer ready at {url}\n{how}\nKeep this window open. Press Ctrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
