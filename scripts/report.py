"""Build runs/<machine>/REPORT.md (every table) and its figures from that machine's runs, plus
runs/COMPARISON.md: throughput of every run on every machine that has results.

The headline summary lives in README.md.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

# Reference palette, light mode: categorical slots 1-4 in fixed order, text and surface tokens.
SERIES = {"mlx": "#2a78d6", "torch": "#eb6834", "mlx-tf32": "#1baf7a", "torch-tf32": "#eda100"}
LABEL = {"mlx": "MLX (mlx-lm)", "torch": "PyTorch", "mlx-tf32": "MLX, TF32", "torch-tf32": "PyTorch, TF32"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

PAIRS = {"fp32": ("mlx-fp32", "torch-fp32"), "bf16": ("mlx-bf16", "torch-bf16")}
TF32 = {"mlx-fp32-tf32": "mlx-fp32", "torch-fp32-tf32": "torch-fp32"}  # TF32 run -> its true-fp32 run
FIG = C.RUNS / "figures"


def load_run(runs_dir, name):
    d = runs_dir / name
    if not (d / "summary.json").exists():
        return None
    return {
        "config": json.loads((d / "config.json").read_text()),
        "summary": json.loads((d / "summary.json").read_text()),
        "log": [json.loads(line) for line in open(d / "log.jsonl")],
    }


def load_json(name):
    p = C.RUNS / name
    return json.loads(p.read_text()) if p.exists() else None


def key(run_name):
    fw = run_name.split("-")[0]
    return f"{fw}-tf32" if run_name.endswith("-tf32") else fw


def precision(run_name):
    return "bf16" if "bf16" in run_name else "fp32, TF32 matmuls" if run_name.endswith("tf32") else "fp32"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_losses(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, (prec, names) in zip(axes, PAIRS.items()):
        style(ax)
        shown = [n for n in names + (tuple(TF32) if prec == "fp32" else ()) if runs.get(n)]
        for n in shown:
            log = runs[n]["log"]
            ax.plot([r["step"] for r in log], [r["loss"] for r in log], color=SERIES[key(n)], linewidth=2,
                    label=LABEL[key(n)], alpha=0.9)
        ax.set_title(f"Training loss, {prec}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Optimizer step", color=INK_2, fontsize=9)
        ax.set_ylabel("Loss (nats/token)", color=INK_2, fontsize=9)
        if len(shown) >= 2:
            ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "loss_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_loss_diff(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3), facecolor=SURFACE)
    for ax, (prec, (a, b)) in zip(axes, PAIRS.items()):
        style(ax)
        if runs.get(a) and runs.get(b):
            la, lb = runs[a]["log"], runs[b]["log"]
            n = min(len(la), len(lb))
            ax.plot(range(n), [la[i]["loss"] - lb[i]["loss"] for i in range(n)], color=INK_2, linewidth=1.5)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(f"Loss difference, MLX minus PyTorch, {prec}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Optimizer step", color=INK_2, fontsize=9)
        ax.set_ylabel("Δ loss", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "loss_diff.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_throughput(runs):
    rows = [n for n in C.ALL_RUNS if runs.get(n)]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1), facecolor=SURFACE)
    style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    vals = [runs[n]["summary"]["tokens_per_s"] for n in rows]
    y = list(range(len(rows)))[::-1]
    ax.barh(y, vals, color=[SERIES[key(n)] for n in rows], height=0.6, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:,.0f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [f"{LABEL[key(n).split('-')[0]]}, {precision(n)}" for n in rows], color=INK)
    ax.set_xlabel("Training throughput, tokens/s (median step, after warmup)", color=INK_2, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title(f"Training throughput, {C.MACHINE}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "throughput.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt(v, spec):
    return "–" if v is None else format(v, spec)


def machine_report():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = {n: load_run(C.RUNS, n) for n in C.ALL_RUNS}
    present = [n for n in C.ALL_RUNS if runs[n]]
    plot_losses(runs)
    plot_loss_diff(runs)
    plot_throughput(runs)

    cfg = runs[present[0]]["config"]
    versions = {}
    for n in present:
        versions.update({k: runs[n]["config"][k] for k in ("mlx", "mlx_lm", "torch", "transformers", "cuda") if k in runs[n]["config"]})
    hw = cfg.get("gpu") or cfg.get("chip")
    out = [
        f"# Full results: {C.MACHINE}", "",
        "Generated by `scripts/report.py` from this directory. Summary and method: [README.md](../../README.md).", "",
        f"- {hw}, {cfg['memory_gb']:.0f} GB; {cfg['os']}; Python {cfg['python']}; "
        + ", ".join(f"{k} {v}" for k, v in versions.items()),
        f"- {cfg['n_steps']} optimizer steps × {cfg['tokens_per_step']:,} tokens "
        f"= {cfg['n_steps'] * cfg['tokens_per_step'] / 1e6:.1f}M training tokens per run; "
        f"peak LR {cfg['peak_lr']}, warmup {cfg['warmup_frac']:.0%}, cosine to {cfg['min_lr_frac']:.0%}; "
        f"AdamW β=({cfg['beta1']}, {cfg['beta2']}), eps {cfg['eps']}, wd {cfg['weight_decay']}; clip {cfg['max_grad_norm']}",
        "",
        "## Speed",
        "",
        "| Run | micro-batch × accum | tokens/s | median step (s) | p90 step (s) | first step (s) | peak memory (GB) | swap Δ (MB) | wall time (min) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n in present:
        s, c = runs[n]["summary"], runs[n]["config"]
        wall = sum(x["step_time"] for x in runs[n]["log"]) / 60
        out.append(
            f"| {n} | {c['micro_batch']} × {c['grad_accum']} | {s['tokens_per_s']:,.0f} | {s['step_time_median_s']:.2f} | "
            f"{s['step_time_p90_s']:.2f} | {s['first_step_s']:.1f} | {s['peak_footprint_gb']:.1f} | "
            f"{s['swap_used_mb_end'] - c['swap_used_mb_start']:+.0f} | {wall:.0f} |"
        )
    out += ["", "Peak memory: on macOS the process's physical footprint (CPU + unified GPU memory) as the OS "
            "accounts it; on Linux the GPU memory nvidia-smi attributes to the process. Either way, the same "
            "measurement for every framework on the machine.", ""]
    for prec, (a, b) in PAIRS.items():
        if runs[a] and runs[b]:
            ratio = runs[a]["summary"]["tokens_per_s"] / runs[b]["summary"]["tokens_per_s"]
            out.append(f"- **{prec}: MLX / PyTorch throughput = {ratio:.2f}×**")
    for t, base in TF32.items():
        if runs[t] and runs[base]:
            out.append(f"- {t} vs {base}: {runs[t]['summary']['tokens_per_s'] / runs[base]['summary']['tokens_per_s']:.2f}×")
    out += ["", "![throughput](figures/throughput.png)", ""]

    out += ["## Loss agreement", "", "| Pair | steps | final loss MLX | final loss PyTorch | max per-step Δ loss | mean per-step Δ loss |",
            "|---|---|---|---|---|---|"]
    for prec, (a, b) in PAIRS.items():
        if runs[a] and runs[b]:
            la, lb = runs[a]["log"], runs[b]["log"]
            n = min(len(la), len(lb))
            d = [abs(la[i]["loss"] - lb[i]["loss"]) for i in range(n)]
            out.append(f"| {prec} | {n} | {la[-1]['loss']:.4f} | {lb[-1]['loss']:.4f} | {max(d):.4f} | {sum(d) / n:.4f} |")
    out += ["", "![loss curves](figures/loss_curves.png)", "", "![loss difference](figures/loss_diff.png)", ""]

    pm = load_json("parity_model.json")
    if pm:
        out += ["## Matching check (fp32, untrained model, 2 held-out sequences)", "", "| Check | Value |", "|---|---|"]
        for k in ("loss_abs_diff", "grad_norm_rel_diff", "grad_rel_diff_median", "grad_rel_diff_max",
                  "grad_rel_diff_max_param", "logits_max_abs_diff", "logits_mean_abs_diff", "argmax_agreement"):
            v = pm[k]
            out.append(f"| {k} | {v if isinstance(v, str) else f'{v:.3g}'} |")
        out.append("")

    probe = load_json("probe_summary.json")
    if probe:
        out += ["## Memory / micro-batch probe (3 steps each)", "",
                "| Setup | micro-batch | ok | tokens/s (step 3) | peak memory (GB) | framework-reported (GB) | swap Δ (MB) |",
                "|---|---|---|---|---|---|---|"]
        for p in probe:
            out.append(f"| {p['setup']} | {p['micro_batch']} | {'yes' if p['ok'] else 'no'} | {fmt(p['tokens_per_s_last'], ',.0f')} | "
                       f"{fmt(p['peak_footprint_gb'], '.1f')} | {fmt(p['framework_mem_gb'], '.1f')} | {fmt(p['swap_delta_mb'], '+.0f')} |")
        out += ["", "Framework-reported memory: MLX = `mx.get_peak_memory()` (true peak of live buffers); "
                "PyTorch CUDA = `torch.cuda.max_memory_allocated()` (true peak); PyTorch MPS = "
                "`torch.mps.driver_allocated_memory()` at step end (includes its cache, not a true peak).", ""]

    conv = load_json("convert.json")
    cold = load_json("convert_cold_load.json") or {}
    if conv:
        out += ["## Conversion to MLX (bf16)", "",
                "| Run | save (s) | mlx_lm.convert (s) | load, warm (s) | load, cold (s) | size (GB) |", "|---|---|---|---|---|---|"]
        for n, r in conv.items():
            out.append(f"| {n} | {fmt(r['save_s'], '.1f')} | {r['convert_s']:.1f} | {r['load_warm_s']:.2f} | "
                       f"{fmt(cold.get(n), '.2f')} | {r['size_gb']:.2f} |")
        out += ["", "Torch → MLX transfer check (same checkpoint, fp32, 2 held-out sequences):", "",
                "| Run | logits max Δ | logits mean Δ | loss, PyTorch fp32 | loss, MLX fp32 | loss, converted MLX bf16 |",
                "|---|---|---|---|---|---|"]
        for n, r in conv.items():
            t = r.get("transfer_check")
            if t:
                out.append(f"| {n} | {t['logits_max_abs_diff']:.2g} | {t['logits_mean_abs_diff']:.2g} | "
                           f"{t['loss_torch_fp32']:.4f} | {t['loss_mlx_fp32']:.4f} | {t['loss_mlx_converted_bf16']:.4f} |")
        out.append("")

    ev = load_json("eval.json")
    if ev:
        out += ["## Perplexity (all models evaluated in mlx-lm, bf16)", "",
                f"| Model | arXiv held-out ({C.load_split('val_arxiv').shape[0] * C.SEQ_LEN:,} tokens) | wikitext-103 test |",
                "|---|---|---|"]
        for n, r in ev.items():
            out.append(f"| {n} | {r['val_arxiv']['ppl']:.3f} | {r['val_wikitext']['ppl']:.3f} |")
        out.append("")

    (C.RUNS / "REPORT.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def comparison():
    """Throughput of every run on every machine with results: runs/COMPARISON.md."""
    machines = sorted(d for d in (C.ROOT / "runs").iterdir() if d.is_dir() and any(d.glob("*/summary.json")))
    table = {m.name: {n: load_run(m, n) for n in C.ALL_RUNS} for m in machines}
    names = [n for n in C.ALL_RUNS if any(table[m][n] for m in table)]
    out = ["# Training throughput across machines", "",
           "Tokens/s (median step after warmup). Same model, data, order and hyperparameters everywhere; "
           "per-machine details in `runs/<machine>/REPORT.md`.", "",
           "| Run | " + " | ".join(table) + " |", "|---|" + "---|" * len(table)]
    for n in names:
        cells = [f"{r['summary']['tokens_per_s']:,.0f} (mb {r['config']['micro_batch']})" if (r := table[m][n]) else "–"
                 for m in table]
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    out += ["", "Final training loss (should agree across machines for the same precision):", "",
            "| Run | " + " | ".join(table) + " |", "|---|" + "---|" * len(table)]
    for n in names:
        out.append(f"| {n} | " + " | ".join(f"{r['summary']['final_loss']:.4f}" if (r := table[m][n]) else "–" for m in table) + " |")
    (C.ROOT / "runs" / "COMPARISON.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    machine_report()
    comparison()
