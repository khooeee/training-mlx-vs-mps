"""Convert every trained checkpoint to a bf16 MLX model and time it; check torch->MLX transfer.

  convert.py              convert all runs (timed), time warm loads, run the transfer check
  convert.py cold         time cold loads only: `sudo purge` before each load (run this yourself;
                          sudo asks for your password once)

Torch checkpoints go save_pretrained -> mlx_lm.convert; MLX checkpoints go through the same
convert command so all models end up in an identical format and dtype.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import common as C

os.environ.setdefault("MLX_ENABLE_TF32", "0")  # see train_mlx.py

RUNS = ["mlx-fp32", "torch-fp32", "mlx-bf16", "torch-bf16", "mlx-fp32-tf32"]
CONVERT = str(C.ROOT / ".venv" / "bin" / "mlx_lm.convert")
LOAD_SNIPPET = (
    "import time, mlx.core as mx, mlx_lm; t = time.perf_counter(); "
    "m, _ = mlx_lm.load({path!r}); mx.eval(m.parameters()); print(time.perf_counter() - t)"
)


def time_load(path):
    out = subprocess.run([sys.executable, "-c", LOAD_SNIPPET.format(path=str(path))],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip().splitlines()[-1])


def convert(run):
    src, dst = C.CKPT / run, C.CKPT / f"{run}-mlx"
    if dst.exists():
        shutil.rmtree(dst)
    t0 = time.perf_counter()
    subprocess.run([CONVERT, "--hf-path", str(src), "--mlx-path", str(dst), "--dtype", "bfloat16"],
                   check=True, capture_output=True)
    convert_s = time.perf_counter() - t0
    summary = json.loads((C.RUNS / run / "summary.json").read_text())
    return {
        "save_s": summary.get("save_s"),
        "convert_s": convert_s,
        "load_warm_s": time_load(dst),
        "size_gb": sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 1e9,
    }


def transfer_check(run):
    """fp32 logits of the torch checkpoint in torch vs the same checkpoint loaded by mlx-lm."""
    import mlx.core as mx
    import mlx_lm
    import torch

    import train_mlx as M
    import train_torch as T
    from transformers import AutoModelForCausalLM

    rows = np.asarray(C.load_split("val_arxiv")[:2])
    x, y = rows[:, :-1].astype(np.int32), rows[:, 1:].astype(np.int32)

    t_model = AutoModelForCausalLM.from_pretrained(C.CKPT / run, dtype=torch.float32).to(T.DEVICE).eval()
    with torch.no_grad():
        t_logits = t_model(input_ids=T.to_device(x), use_cache=False).logits.float()
        t_loss = torch.nn.functional.cross_entropy(t_logits.view(-1, t_logits.size(-1)), T.to_device(y).view(-1)).item()
    t_logits = t_logits.cpu().numpy()
    del t_model

    m_model, _ = mlx_lm.load(str(C.CKPT / run))
    m_model.set_dtype(mx.float32)
    m_logits = np.array(m_model(mx.array(x)).astype(mx.float32))
    m_loss = M.loss_fn(m_model, mx.array(x), mx.array(y)).item()

    conv, _ = mlx_lm.load(str(C.CKPT / f"{run}-mlx"))  # the bf16 model actually used downstream
    conv_loss = M.loss_fn(conv, mx.array(x), mx.array(y)).item()
    return {
        "logits_max_abs_diff": float(np.abs(m_logits - t_logits).max()),
        "logits_mean_abs_diff": float(np.abs(m_logits - t_logits).mean()),
        "loss_torch_fp32": t_loss,
        "loss_mlx_fp32": m_loss,
        "loss_mlx_converted_bf16": conv_loss,
    }


def main():
    results = {}
    for run in RUNS:
        if not (C.CKPT / run).exists():
            continue
        results[run] = convert(run)
        if run.startswith("torch"):
            results[run]["transfer_check"] = transfer_check(run)
        print(run, json.dumps(results[run]), flush=True)
    (C.RUNS / "convert.json").write_text(json.dumps(results, indent=2))


def cold():
    results = {}
    for run in RUNS:
        dst = C.CKPT / f"{run}-mlx"
        if dst.exists():
            subprocess.run(["sudo", "purge"], check=True)
            results[run] = time_load(dst)
            print(f"{run}: cold load {results[run]:.2f}s", flush=True)
    (C.RUNS / "convert_cold_load.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    cold() if sys.argv[1:] == ["cold"] else main()
