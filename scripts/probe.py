"""Memory/throughput probe: a few steps per (setup, micro-batch) to pick the shared micro-batch
and the run length. Each probe is a separate process so an OOM can't take down the rest.
"""

import json
import os
import subprocess
import sys

import common as C

SETUPS = {  # name -> (script, dtype, extra env)
    "mlx-fp32": ("train_mlx.py", "float32", {"MLX_ENABLE_TF32": "0"}),
    "torch-fp32": ("train_torch.py", "float32", {}),
    "mlx-bf16": ("train_mlx.py", "bfloat16", {}),
    "torch-bf16": ("train_torch.py", "bfloat16", {}),
    "mlx-fp32-tf32": ("train_mlx.py", "float32", {"MLX_ENABLE_TF32": "1"}),
}
MICRO_BATCHES = [1, 2, 4, 8]
STEPS = 3
FOOTPRINT_LIMIT_GB = 27.0  # ~ the 26.8 GB recommended GPU working set
SWAP_LIMIT_MB = 256  # any real swap growth makes timings meaningless


def probe(setup, mb):
    script, dtype, env = SETUPS[setup]
    name = f"probe-{setup}-mb{mb}"
    proc = subprocess.run(
        [sys.executable, str(C.ROOT / "scripts" / script), "--dtype", dtype, "--micro-batch", str(mb),
         "--max-steps", str(STEPS), "--total-steps", "1000", "--run-name", name],
        env={**os.environ, **env}, capture_output=True, text=True,
    )
    log = C.RUNS / name / "log.jsonl"
    recs = [json.loads(line) for line in open(log)] if log.exists() else []
    summary_path = C.RUNS / name / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    ok = proc.returncode == 0 and len(recs) == STEPS
    res = {
        "setup": setup, "micro_batch": mb, "ok": ok,
        "step_times": [round(r["step_time"], 2) for r in recs],
        "tokens_per_s_last": recs[-1]["tokens"] / recs[-1]["step_time"] if recs else None,
        "peak_footprint_gb": max(r["peak_footprint"] for r in recs) / 1e9 if recs else None,
        "framework_mem_gb": max(r["framework_peak_mem"] for r in recs) / 1e9 if recs else None,
        "swap_delta_mb": (summary.get("swap_used_mb_end", 0) - json.loads((C.RUNS / name / "config.json").read_text())["swap_used_mb_start"]) if summary else None,
        "error": None if ok else proc.stderr.strip().splitlines()[-1:],
    }
    print(json.dumps(res), flush=True)
    return res


def main():
    results = []
    for setup in SETUPS:
        for mb in MICRO_BATCHES:
            r = probe(setup, mb)
            results.append(r)
            if not r["ok"] or r["peak_footprint_gb"] > FOOTPRINT_LIMIT_GB or r["swap_delta_mb"] > SWAP_LIMIT_MB:
                break
    (C.RUNS / "probe_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
