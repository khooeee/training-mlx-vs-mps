"""Where does a training step's time go? Times forward, forward+backward and the optimizer
update separately (synchronizing between phases) for each framework and dtype, at micro-batch 1.

  .venv/bin/python scripts/diag_step_phases.py                # all installed frameworks
  .venv/bin/python scripts/diag_step_phases.py torch          # one framework

Diagnostic only: the trainers don't sync between phases, so these don't add up to a step time.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

import common as C

os.environ.setdefault("MLX_ENABLE_TF32", "0")  # see train_mlx.py
N = 5  # timed repetitions per phase, after one warmup


def batch():
    rows = np.asarray(C.load_split("train")[:1])
    return rows[:, :-1].astype(np.int32), rows[:, 1:].astype(np.int32)


def timed(fn, sync):
    fn()
    sync()
    t = time.perf_counter()
    for _ in range(N):
        fn()
    sync()
    return (time.perf_counter() - t) / N


def torch_phases(dtype, attn):
    import torch
    from transformers import AutoModelForCausalLM

    import train_torch as T

    model = AutoModelForCausalLM.from_pretrained(C.BASE_MODEL, dtype=getattr(torch, dtype), attn_implementation=attn)
    model = model.to(T.DEVICE).train()
    opt = T.make_optimizer(C.TrainConfig(dtype=dtype), model)
    x, y = (T.to_device(a) for a in batch())
    hidden = None

    def fwd():
        with torch.no_grad():
            T.loss_fn(model, x, y)

    def fwd_bwd():
        T.loss_fn(model, x, y).backward()

    def backbone_fwd():
        nonlocal hidden
        with torch.no_grad():
            hidden = model.model(input_ids=x, use_cache=False).last_hidden_state

    def head_loss_fwd_bwd():  # LM head (151,936-way) + fp32 cross-entropy on its own
        h = hidden.detach().requires_grad_()
        logits = model.lm_head(h).float()
        torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).backward()

    res = {"fwd": timed(fwd, T.sync), "fwd+bwd": timed(fwd_bwd, T.sync), "optimizer": timed(opt.step, T.sync)}
    backbone_fwd()
    res["backbone fwd"] = timed(backbone_fwd, T.sync)
    res["lm_head+loss fwd+bwd"] = timed(head_loss_fwd_bwd, T.sync)
    del model, opt
    getattr(torch, T.DEVICE).empty_cache()
    return res


def mlx_phases(dtype):
    import mlx.core as mx
    import mlx.nn as nn

    import train_mlx as M

    model, _, _ = M.build(dtype)
    cfg = C.TrainConfig(dtype=dtype)
    opt = M.make_optimizer(cfg, model)
    x, y = (mx.array(a) for a in batch())
    loss_and_grad = nn.value_and_grad(model, M.loss_fn)
    grads = None

    def fwd():
        mx.eval(M.loss_fn(model, x, y))

    def fwd_bwd():
        nonlocal grads
        _, grads = loss_and_grad(model, x, y)
        mx.eval(grads)

    def update():
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    res = {"fwd": timed(fwd, lambda: None), "fwd+bwd": timed(fwd_bwd, lambda: None), "optimizer": timed(update, lambda: None)}
    del model, opt, grads
    mx.clear_cache()
    return res


def main():
    fws = sys.argv[1:] or [fw for fw in ("mlx", "torch") if importlib.util.find_spec(fw)]
    out = {}
    for dtype in ("float32", "bfloat16"):
        if "mlx" in fws:
            out[f"mlx {dtype}"] = mlx_phases(dtype)
            print(f"mlx {dtype}", json.dumps({k: round(v, 3) for k, v in out[f'mlx {dtype}'].items()}), flush=True)
        if "torch" in fws:
            for attn in ("sdpa", "eager"):
                out[f"torch {dtype} {attn}"] = torch_phases(dtype, attn)
                print(f"torch {dtype} {attn}", json.dumps({k: round(v, 3) for k, v in out[f'torch {dtype} {attn}'].items()}), flush=True)
    C.RUNS.mkdir(parents=True, exist_ok=True)
    (C.RUNS / "diag_step_phases.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
