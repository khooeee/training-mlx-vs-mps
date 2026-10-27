"""Matmul precision and throughput per framework and dtype, against a float64 CPU reference.

Run this first on any new machine: it is how MLX's TF32-by-default on M5 showed up
(fp32 relative error ~1e-3 instead of ~1e-7). Skips any framework that isn't installed.

  .venv/bin/python scripts/matmul_check.py
"""

import json
import os
import time

import numpy as np

M, K, N = 4096, 1024, 3072  # a Qwen3-0.6B MLP-sized matmul at 4 x 1024 tokens
ITERS = 50


def report(name, out, seconds, ref):
    err = float(np.linalg.norm(out.astype(np.float64) - ref) / np.linalg.norm(ref))
    rec = {"backend": name, "rel_err": err, "tflops": 2 * M * K * N / seconds / 1e12}
    print(json.dumps(rec))
    return rec


def check_torch(A, B, ref):
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if device is None:
        return
    sync = torch.cuda.synchronize if device == "cuda" else torch.mps.synchronize
    for prec in ("highest", "high"):  # "high" allows TF32 where the backend supports it
        torch.set_float32_matmul_precision(prec)
        for dtype in (torch.float32, torch.bfloat16):
            if dtype == torch.bfloat16 and prec != "highest":
                continue
            a, b = torch.from_numpy(A).to(device, dtype), torch.from_numpy(B).to(device, dtype)
            c = a @ b
            sync()
            t = time.perf_counter()
            for _ in range(ITERS):
                c = a @ b
            sync()
            report(f"torch {device} {str(dtype).split('.')[1]} precision={prec}", c.float().cpu().numpy(),
                   (time.perf_counter() - t) / ITERS, ref)
    torch.set_float32_matmul_precision("highest")


def check_mlx(A, B, ref):
    try:
        import mlx.core as mx
    except ImportError:
        return
    for dtype in (mx.float32, mx.bfloat16):
        a, b = mx.array(A).astype(dtype), mx.array(B).astype(dtype)
        c = a @ b
        mx.eval(c)
        t = time.perf_counter()
        for _ in range(ITERS):
            c = a @ b
            mx.eval(c)
        tf32 = os.environ.get("MLX_ENABLE_TF32", "default")
        report(f"mlx {mx.default_device()} {dtype} MLX_ENABLE_TF32={tf32}", np.array(c.astype(mx.float32)),
               (time.perf_counter() - t) / ITERS, ref)


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    print(json.dumps({"backend": "numpy cpu float32", "rel_err": float(np.linalg.norm(A @ B - ref) / np.linalg.norm(ref))}))
    check_torch(A, B, ref)
    check_mlx(A, B, ref)


if __name__ == "__main__":
    main()
