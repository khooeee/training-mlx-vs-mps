"""Memory/throughput probe: a few steps per (setup, micro-batch) to pick the shared micro-batch
and the run length. Each probe is a separate process so an OOM can't take down the rest.
Probes whichever frameworks are installed; results go to runs/<machine>/probe_summary.json.
"""

import importlib.util
import json
import shutil
import subprocess
import sys

import common as C

MICRO_BATCHES = [1, 2, 4, 8]
STEPS = 3
FOOTPRINT_LIMIT_GB = 27.0  # on the M5: ~ the 26.8 GB recommended GPU working set
SWAP_LIMIT_MB = 256  # any real swap growth makes timings meaningless


def setups():
    """name -> (framework, dtype, extra args), in the same order run_all.sh uses."""
    fws = [fw for fw in ("mlx", "torch") if importlib.util.find_spec(fw)]
    out = {}
    for dtype, tag in (("float32", "fp32"), ("bfloat16", "bf16")):
        for fw in fws:
            out[f"{fw}-{tag}"] = (fw, dtype, [])
    for fw in fws:
        if fw == "mlx" or shutil.which("nvidia-smi"):  # PyTorch has TF32 on CUDA only
            out[f"{fw}-fp32-tf32"] = (fw, "float32", ["--tf32"])
    return out


def probe(setup, fw, dtype, extra, mb):
    name = f"probe-{setup}-mb{mb}"
    proc = subprocess.run(
        [sys.executable, str(C.ROOT / "scripts" / f"train_{fw}.py"), "--dtype", dtype, "--micro-batch", str(mb),
         "--max-steps", str(STEPS), "--total-steps", "1000", "--run-name", name, *extra],
        capture_output=True, text=True,
    )
    log = C.RUNS / name / "log.jsonl"
    recs = [json.loads(line) for line in open(log)] if log.exists() else []
    summary_path = C.RUNS / name / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    ok = proc.returncode == 0 and len(recs) == STEPS
    swap_start = json.loads((C.RUNS / name / "config.json").read_text())["swap_used_mb_start"] if recs else None
    res = {
        "setup": setup, "micro_batch": mb, "ok": ok,
        "step_times": [round(r["step_time"], 2) for r in recs],
        "tokens_per_s_last": recs[-1]["tokens"] / recs[-1]["step_time"] if recs else None,
        "peak_footprint_gb": max(r["peak_footprint"] or 0 for r in recs) / 1e9 if recs else None,
        "framework_mem_gb": max(r["framework_peak_mem"] for r in recs) / 1e9 if recs else None,
        "swap_delta_mb": summary["swap_used_mb_end"] - swap_start if summary else None,
        "error": None if ok else proc.stderr.strip().splitlines()[-1:],
    }
    print(json.dumps(res), flush=True)
    return res


def main():
    results = []
    for setup, (fw, dtype, extra) in setups().items():
        for mb in MICRO_BATCHES:
            r = probe(setup, fw, dtype, extra, mb)
            results.append(r)
            if not r["ok"] or r["peak_footprint_gb"] > FOOTPRINT_LIMIT_GB or r["swap_delta_mb"] > SWAP_LIMIT_MB:
                break
    (C.RUNS / "probe_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
