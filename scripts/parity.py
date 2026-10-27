"""Check that MLX and PyTorch compute the same thing before any long run.

  parity.py model         fp32 forward (logits, loss) and backward (per-parameter grads)
                          on two held-out sequences, using each trainer's own build/loss code
  parity.py runs A B      compare per-step loss and grad norm of two training runs
                          (optimizer + LR schedule + accumulation check)
"""

import json
import os
import sys

import numpy as np

os.environ.setdefault("MLX_ENABLE_TF32", "0")  # see train_mlx.py

import common as C


def check_model():
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    from mlx.utils import tree_flatten

    import train_mlx as M
    import train_torch as T

    rows = np.asarray(C.load_split("val_arxiv")[:2])
    x, y = rows[:, :-1].astype(np.int32), rows[:, 1:].astype(np.int32)

    m_model, _, _ = M.build("float32")
    t_model = T.build("float32")

    m_loss, m_grads = nn.value_and_grad(m_model, M.loss_fn)(m_model, mx.array(x), mx.array(y))
    m_logits = m_model(mx.array(x))
    mx.eval(m_loss, m_grads, m_logits)
    m_logits = np.array(m_logits)
    m_grads = {k: np.array(v) for k, v in tree_flatten(m_grads)}

    t_loss = T.loss_fn(t_model, T.to_device(x), T.to_device(y))
    t_loss.backward()
    with torch.no_grad():
        t_logits = t_model(input_ids=T.to_device(x), use_cache=False).logits.float().cpu().numpy()
    t_grads = {k: p.grad.float().cpu().numpy() for k, p in t_model.named_parameters()}

    assert m_grads.keys() == t_grads.keys(), (
        f"param names differ: only mlx {sorted(m_grads.keys() - t_grads.keys())[:5]}, "
        f"only torch {sorted(t_grads.keys() - m_grads.keys())[:5]}"
    )
    rel = {k: float(np.linalg.norm(m_grads[k] - t_grads[k]) / (np.linalg.norm(t_grads[k]) + 1e-30)) for k in t_grads}
    worst = max(rel, key=rel.get)
    m_gnorm = float(np.sqrt(sum(np.sum(g.astype(np.float64) ** 2) for g in m_grads.values())))
    t_gnorm = float(np.sqrt(sum(np.sum(g.astype(np.float64) ** 2) for g in t_grads.values())))

    res = {
        "logits_max_abs_diff": float(np.abs(m_logits - t_logits).max()),
        "logits_mean_abs_diff": float(np.abs(m_logits - t_logits).mean()),
        "logits_abs_max": float(np.abs(t_logits).max()),
        "argmax_agreement": float((m_logits.argmax(-1) == t_logits.argmax(-1)).mean()),
        "loss_mlx": m_loss.item(),
        "loss_torch": t_loss.item(),
        "loss_abs_diff": abs(m_loss.item() - t_loss.item()),
        "grad_norm_mlx": m_gnorm,
        "grad_norm_torch": t_gnorm,
        "grad_norm_rel_diff": abs(m_gnorm - t_gnorm) / t_gnorm,
        "grad_rel_diff_median": float(np.median(list(rel.values()))),
        "grad_rel_diff_max": rel[worst],
        "grad_rel_diff_max_param": worst,
        "n_params_compared": len(rel),
    }
    res["pass"] = bool(
        res["loss_abs_diff"] < 1e-4
        and res["grad_norm_rel_diff"] < 1e-3
        and res["grad_rel_diff_max"] < 1e-2
        and res["argmax_agreement"] > 0.999
    )
    C.RUNS.mkdir(exist_ok=True)
    (C.RUNS / "parity_model.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def read_log(name):
    return [json.loads(line) for line in open(C.RUNS / name / "log.jsonl")]


def check_runs(a, b):
    ra, rb = read_log(a), read_log(b)
    n = min(len(ra), len(rb))
    loss_diff = [abs(ra[i]["loss"] - rb[i]["loss"]) for i in range(n)]
    gn_rel = [abs(ra[i]["grad_norm"] - rb[i]["grad_norm"]) / rb[i]["grad_norm"] for i in range(n)]
    res = {
        "runs": [a, b],
        "steps": n,
        "loss": [[ra[i]["loss"], rb[i]["loss"]] for i in range(n)],
        "loss_abs_diff_max": max(loss_diff),
        "grad_norm_rel_diff_max": max(gn_rel),
        "lr_match": all(abs(ra[i]["lr"] - rb[i]["lr"]) < 1e-12 for i in range(n)),
    }
    print(json.dumps(res, indent=2))
    (C.RUNS / f"parity_{a}_vs_{b}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "model":
        check_model()
    else:
        check_runs(sys.argv[2], sys.argv[3])
