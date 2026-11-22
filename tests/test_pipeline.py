"""The unattended pipeline: experiment batches resume after a shutdown, and fitted settings are
written into the config without losing its comments."""
import importlib.util
import json
import shutil
from pathlib import Path

from brain_model.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_resume_skips_finished_runs(tmp_path):
    rb = _script("run_batch")
    base = load_config(ROOT / "configs" / "hcp.yaml")
    todo = rb.jobs(base, ["44", "6mp"], [1, 2])
    finished = [{"experiment": e, "condition": c, "start_area": a, "seed": cfg["seed"], "benefit": 0.1}
                for e, c, a, cfg in todo[:5]]
    partial = tmp_path / "results_partial.jsonl"
    partial.write_text("\n".join(json.dumps(r) for r in finished) + '\n{"experiment": "ha', encoding="utf-8")

    rows = rb.load_partial(partial)  # the half-written last line (power cut) is ignored
    assert len(rows) == 5
    done = {rb.job_key(r) for r in rows}
    remaining = [j for j in todo if rb.job_key(j) not in done]
    assert len(remaining) == len(todo) - 5
    assert rb.job_key(todo[0]) == rb.job_key(finished[0])


def test_apply_settings_rewrites_only_the_fitted_numbers(tmp_path):
    fit = _script("fit_to_fmri")
    cfg_file = tmp_path / "hcp.yaml"
    shutil.copy(ROOT / "configs" / "hcp.yaml", cfg_file)
    before = load_config(cfg_file)
    g = {"leak_l": 0.111, "gain_l": 0.222, "noise_l": 0.033,
         "leak_r": 0.444, "gain_r": 0.555, "noise_r": 0.066, "strength": 0.777}
    fit.apply_settings(cfg_file, g, Path("results/fit_main"))

    after = load_config(cfg_file)
    assert after["left_hemisphere"]["dynamics"] == {"leak": 0.111, "gain": 0.222, "noise": 0.033}
    assert after["right_hemisphere"]["dynamics"] == {"leak": 0.444, "gain": 0.555, "noise": 0.066}
    assert after["corpus_callosum"]["strength"] == 0.777
    for key in ("dynamics", "lesion", "anatomy"):  # everything else untouched
        assert after[key] == before[key]
    assert after["left_hemisphere"]["functions"] == before["left_hemisphere"]["functions"]
    assert (tmp_path / "hcp.yaml.bak").exists()
    assert "# REAL-BRAIN experiment" in cfg_file.read_text(encoding="utf-8")
