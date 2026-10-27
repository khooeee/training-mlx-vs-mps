"""Full-weight continued pretraining of Qwen3-0.6B-Base with MLX (mlx-lm model code).

Mirrors train_torch.py step for step; see common.py for the shared config, data
order and LR schedule.
"""

import json
import os
import sys
import time
from functools import partial

# On M5 GPUs MLX runs float32 matmuls as TF32 by default (~2.5x faster, ~1e-3 relative
# error). PyTorch MPS has no TF32 mode, so use true fp32 unless --tf32 is passed. MLX reads
# the variable when it loads, so this has to happen before the import.
os.environ["MLX_ENABLE_TF32"] = "1" if "--tf32" in sys.argv else "0"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_map
from mlx_lm.utils import maybe_set_recommended_wired_limit, save

import common as C


def build(dtype):
    model, tokenizer, config = mlx_lm.load(str(C.BASE_MODEL), return_config=True)
    model.set_dtype(getattr(mx, dtype))
    mx.eval(model.parameters())
    return model, tokenizer, config


def loss_fn(model, x, y):
    logits = model(x).astype(mx.float32)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def make_optimizer(cfg, model):
    opt = optim.AdamW(
        learning_rate=cfg.peak_lr,
        betas=[cfg.beta1, cfg.beta2],
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        bias_correction=True,  # MLX defaults to False; PyTorch always corrects
    )
    opt.init(model.trainable_parameters())
    return opt


def make_step_fns(cfg, model, optimizer):
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def micro_step(x, y, acc):
        loss, grad = loss_and_grad(model, x, y)
        if acc is not None:
            grad = tree_map(mx.add, grad, acc)
        return loss, grad

    @partial(mx.compile, inputs=state, outputs=state)
    def update(grad):
        grad = tree_map(lambda g: g / cfg.grad_accum, grad)
        grad, norm = optim.clip_grad_norm(grad, cfg.max_grad_norm)
        optimizer.update(model, grad)
        return norm

    return micro_step, update


def clear_cache():
    # mlx-lm's trainer default (clear_cache_threshold=0): drop the allocator cache every iteration.
    if mx.get_cache_memory() > 0:
        mx.clear_cache()


def main():
    args, cfg = C.parse_args("mlx")
    n_steps = args.max_steps or cfg.total_steps
    maybe_set_recommended_wired_limit()  # mlx-lm's trainer does this by default

    model, tokenizer, config = build(cfg.dtype)
    optimizer = make_optimizer(cfg, model)
    micro_step, update = make_step_fns(cfg, model, optimizer)
    batches = C.Batches(cfg, n_steps)
    meter = C.MemoryMeter()

    log = C.RunLog(args.run_name, {
        "framework": "mlx", "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
        "MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"],
        **C.system_info(), **vars(cfg), "grad_accum": cfg.grad_accum, "n_steps": n_steps,
        "swap_used_mb_start": C.swap_used_mb(), "thermal_start": C.thermal_state(),
    })

    records = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        acc, losses = None, []
        for micro in range(cfg.grad_accum):
            x, y = batches.get(step, micro)
            loss, acc = micro_step(mx.array(x), mx.array(y), acc)
            mx.eval(loss, acc)
            clear_cache()
            losses.append(loss)
        lr = C.lr_at(cfg, step)
        optimizer.learning_rate = lr
        norm = update(acc)
        mx.eval(model.parameters(), optimizer.state, norm)
        clear_cache()
        step_time = time.perf_counter() - t0

        cur_mem, peak_mem = meter.sample()
        rec = {
            "step": step, "loss": sum(l.item() for l in losses) / len(losses), "lr": lr,
            "grad_norm": norm.item(), "step_time": step_time, "tokens": cfg.tokens_per_step,
            "footprint": cur_mem, "peak_footprint": peak_mem, "framework_peak_mem": mx.get_peak_memory(),
            "thermal": C.thermal_level(),
        }
        records.append(rec)
        log.log(**rec)
        print(f"step {step:4d} loss {rec['loss']:.4f} gnorm {rec['grad_norm']:.3f} "
              f"{step_time:.2f}s {cfg.tokens_per_step / step_time:.0f} tok/s "
              f"peak {peak_mem / 1e9:.1f}GB thermal {rec['thermal']}", flush=True)

    summary = {**C.summarize(records), "swap_used_mb_end": C.swap_used_mb(), "thermal_end": C.thermal_state()}
    if args.save:
        t0 = time.perf_counter()
        save(C.CKPT / args.run_name, C.BASE_MODEL, model, tokenizer, config, donate_model=False)
        summary["save_s"] = time.perf_counter() - t0
    (log.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
