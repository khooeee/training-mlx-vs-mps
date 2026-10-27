"""Full-weight continued pretraining of Qwen3-0.6B-Base with PyTorch (HF transformers model code).

Runs on MPS (Apple Silicon) or CUDA (NVIDIA), whichever is available. Mirrors train_mlx.py
step for step; see common.py for the shared config, data order and LR schedule.
"""

import json
import time

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C

DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def sync():
    torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()


def framework_mem():
    # CUDA: true peak of live tensors. MPS has no peak API: driver allocation at step end,
    # which includes the allocator cache.
    return torch.cuda.max_memory_allocated() if DEVICE == "cuda" else torch.mps.driver_allocated_memory()


def build(dtype):
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, dtype=getattr(torch, dtype), attn_implementation="sdpa"
    ).to(DEVICE)
    model.train()
    return model


def loss_fn(model, x, y):
    logits = model(input_ids=x, use_cache=False).logits.float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


def make_optimizer(cfg, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        fused=True,  # single-kernel update, the closest analog to MLX's compiled update
    )


def to_device(a):
    return torch.from_numpy(a).long().to(DEVICE)


def main():
    args, cfg = C.parse_args("torch")
    n_steps = args.max_steps or cfg.total_steps
    if cfg.tf32 and DEVICE != "cuda":
        raise SystemExit("--tf32: PyTorch has no TF32 mode on MPS (fp32 matmuls are always full precision)")
    torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")

    model = build(cfg.dtype)
    optimizer = make_optimizer(cfg, model)
    batches = C.Batches(cfg, n_steps)
    meter = C.MemoryMeter()

    log = C.RunLog(args.run_name, {
        "framework": "torch", "torch": torch.__version__, "transformers": transformers.__version__,
        "device": DEVICE, "float32_matmul_precision": torch.get_float32_matmul_precision(),
        **({"cuda": torch.version.cuda} if DEVICE == "cuda" else {}),
        **C.system_info(), **vars(cfg), "grad_accum": cfg.grad_accum, "n_steps": n_steps,
        "swap_used_mb_start": C.swap_used_mb(), "thermal_start": C.thermal_state(),
    })

    records = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=DEVICE)
        for micro in range(cfg.grad_accum):
            x, y = batches.get(step, micro)
            loss = loss_fn(model, to_device(x), to_device(y))
            (loss / cfg.grad_accum).backward()
            loss_sum += loss.detach()
        lr = C.lr_at(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync()
        step_time = time.perf_counter() - t0

        cur_mem, peak_mem = meter.sample()
        rec = {
            "step": step, "loss": loss_sum.item() / cfg.grad_accum, "lr": lr,
            "grad_norm": norm.item(), "step_time": step_time, "tokens": cfg.tokens_per_step,
            "footprint": cur_mem, "peak_footprint": peak_mem, "framework_peak_mem": framework_mem(),
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
        model.save_pretrained(C.CKPT / args.run_name)
        AutoTokenizer.from_pretrained(C.BASE_MODEL).save_pretrained(C.CKPT / args.run_name)
        summary["save_s"] = time.perf_counter() - t0
    (log.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
