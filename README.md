# MLX vs PyTorch MPS: full-weight training on Apple Silicon

An apples-to-apples comparison of **full-weight continued pretraining (CPT)** of the same model with
**MLX (mlx-lm)** and **PyTorch on MPS**, on the same Mac. The main question is which one trains faster.
Model quality is a sanity check: both should learn the same amount. It also measures what it costs to
take a PyTorch-trained model into MLX for inference.

- **Model:** [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), all 0.6B parameters trained (no LoRA)
- **Data:** full-text arXiv papers ([Common Pile `arxiv_papers_filtered`](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered), one shard)
- **Machine:** Apple M5 (10-core GPU), 32 GB unified memory, macOS 27.0.1

## Results

_Pending: the overnight runs fill this in. Every table and figure is in [REPORT.md](REPORT.md)._

## What "apples to apples" means here

Everything that isn't the framework is held identical. Each framework still runs the model its normal
way, because that difference is what's being measured.

| Held identical | Allowed to differ (what's being measured) |
|---|---|
| Model and starting weights | Model code: mlx-lm's Qwen3 vs Hugging Face transformers' Qwen3 |
| Token stream: tokenized once, packed into 1024-token rows, same order (seed 0) | Kernels: MLX fused attention/RMSNorm/RoPE + `mx.compile` vs PyTorch eager + SDPA |
| Tokens per step (16,384), number of steps, LR schedule (computed by hand in both) | Optimizer implementation: MLX compiled update vs PyTorch fused AdamW |
| AdamW: β=(0.9, 0.95), eps 1e-8, weight decay 0.1 on all params, bias correction on | Memory management (each framework's allocator) |
| Grad clipping 1.0, fp32 loss, grad accumulation, no grad checkpointing, no dropout | |
| Precision: weights, grads and optimizer state all fp32, or all bf16 | |
| Runs one at a time, alternating frameworks, 60 s cooldown, plugged in | |

Before any long run, `scripts/parity.py` checks that the two implementations compute the same function.
On the untrained model in fp32, with two held-out sequences:

| Check | MLX vs PyTorch |
|---|---|
| Loss | differ by 4.8e-7 (2.352563 vs 2.352563) |
| Global gradient norm | relative difference 1.5e-6 |
| Per-parameter gradients | median relative difference 3.4e-4 (worst 8.5e-3, a layernorm weight) |
| Logits | mean abs difference 7.9e-6, max 3.7e-4 (max logit 34) |
| Next-token argmax | identical at 100% of positions |

## Things that silently break the comparison

These all came up while building this. Each one would have skewed the result without any error.

1. **On M5, MLX runs fp32 matmuls as TF32 by default.** On a 4096×1024×3072 matmul, MLX on the GPU had
   relative error 7.7e-4 (between fp16 and bf16), against 5.7e-7 for numpy, MLX on the CPU and PyTorch MPS.
   That makes it about 2.5× faster (7.6–9.1 vs 3.5 TFLOPS). PyTorch MPS has no TF32 mode;
   `torch.set_float32_matmul_precision` has no effect on MPS. This made the first parity check fail.
   The fp32 comparison sets `MLX_ENABLE_TF32=0`. One extra run (`mlx-fp32-tf32`) measures MLX's default.
2. **mlx-lm's trainer clears MLX's buffer cache every iteration** (`clear_cache_threshold=0` is the default).
   A hand-written MLX loop that doesn't do this reached a 25–26 GB footprint (against 20 GB of live buffers)
   and pushed the machine into swap. `train_mlx.py` matches mlx-lm's default.
3. **MLX's `AdamW` defaults to `bias_correction=False`.** PyTorch always applies bias correction. With it
   turned on, MLX's update formula is the same as PyTorch's: decoupled weight decay first, step count
   incremented before the update.
4. **Hugging Face builds a KV cache during training forward passes** unless you pass `use_cache=False`.
5. **Pure bf16 swallows small updates.** At a CPT-typical LR (2e-5), most Adam updates are smaller than
   bf16's resolution near typical weight magnitudes, so they round away. `(1 - lr·wd)` also rounds to 1, so
   weight decay is a no-op. This hits both frameworks equally, so the speed comparison still holds, but expect
   the bf16 runs to learn less than fp32. Storage dtypes match exactly. Internal op precision can differ:
   PyTorch's fused AdamW does its arithmetic in fp32 before rounding back.
6. **The frameworks' memory counters aren't comparable.** `mx.get_peak_memory()` is a true peak of live
   buffers. PyTorch MPS has no peak API, and `driver_allocated_memory()` includes its cache. The comparable
   number is the process's physical footprint as macOS accounts it (`proc_pid_rusage`,
   `ri_lifetime_max_phys_footprint`), which covers CPU and unified GPU memory alike.
7. **Timing lazy and async execution.** MLX is lazy: each step `mx.eval`s parameters and optimizer state.
   MPS is async: each step ends with `torch.mps.synchronize()`. The first 20 steps (compilation, shader
   caching) are excluded from throughput.

## Setup

Requires [uv](https://docs.astral.sh/uv/). `requirements.txt` is the exact environment that produced the results.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download.py               # model + data at pinned revisions (~1.4 GB)
.venv/bin/python -I scripts/prepare_data.py        # tokenize + pack: 24M train / 525k arXiv val / 298k wikitext tokens
```

## Reproduce

```bash
.venv/bin/python scripts/parity.py model           # fp32 forward/backward agreement -> runs/parity_model.json
.venv/bin/python scripts/probe.py                  # memory + speed per (setup, micro-batch) -> runs/probe_summary.json

# Everything else, one job at a time (~5-6 h on an M5): 5 training runs, conversion, perplexity, report.
# Arguments: fp32 micro-batch, bf16 micro-batch, optimizer steps per run.
caffeinate -is scripts/run_all.sh <fp32_mb> <bf16_mb> <steps>
tail -f runs/logs/run_all.log

# Optional: cold-load timing (flushes the file cache before each load; asks for your password)
.venv/bin/python scripts/convert.py cold && .venv/bin/python scripts/report.py
```

Individual pieces:

```bash
.venv/bin/python scripts/train_mlx.py   --dtype float32 --micro-batch 2 --total-steps 200 --save
.venv/bin/python scripts/train_torch.py --dtype float32 --micro-batch 2 --total-steps 200 --save
.venv/bin/python scripts/parity.py runs mlx-fp32 torch-fp32   # per-step loss / grad-norm agreement
.venv/bin/python scripts/convert.py                           # -> checkpoints/<run>-mlx (bf16), timed
.venv/bin/python scripts/eval_ppl.py                          # perplexity, all models in mlx-lm
.venv/bin/python scripts/report.py                            # -> REPORT.md, figures/
```

## Compare model outputs

All models, including the PyTorch-trained ones after conversion, run in mlx-lm. They are base models, so
give them text to continue rather than questions. `--temp 0` makes decoding greedy and repeatable.
`--ignore-chat-template` stops mlx-lm from wrapping the prompt in chat formatting.

```bash
P="In this paper, we propose a novel method for"
for m in models/Qwen3-0.6B-Base checkpoints/mlx-fp32-mlx checkpoints/torch-fp32-mlx; do
  echo "=== $m"
  .venv/bin/mlx_lm.generate --model "$m" --prompt "$P" --max-tokens 200 --temp 0 --ignore-chat-template
done
```

Or run every prompt in `prompts.txt` side by side:

```bash
.venv/bin/python scripts/compare.py                                    # base vs mlx-fp32 vs torch-fp32
.venv/bin/python scripts/compare.py --models base mlx-bf16 torch-bf16
.venv/bin/python scripts/compare.py --prompt "Theorem 2. Let X be a compact" --max-tokens 200
```

The two trained models have different weights, so their greedy outputs diverge after a few tokens. What
to look for is whether both have picked up arXiv style to a similar degree, compared with the base model.
The perplexity table is the quantitative version of that check.

## Layout

```
scripts/
  download.py      model + data at pinned revisions
  prepare_data.py  tokenize once, pack 1024+1-token rows, split train/val by paper
  common.py        shared config, data order, LR schedule, logging, OS memory/swap/thermal readouts
  train_mlx.py     MLX trainer      } mirror each other step for step
  train_torch.py   PyTorch trainer  }
  parity.py        forward/backward agreement; per-step run agreement
  probe.py         memory + throughput per micro-batch
  run_all.sh       the overnight sequence
  convert.py       timed save -> mlx_lm.convert (bf16) -> load; torch->MLX transfer check; cold loads
  eval_ppl.py      perplexity of every model in mlx-lm
  report.py        REPORT.md + figures/
  compare.py       side-by-side greedy generations
prompts.txt        prompts for compare.py
runs/<run>/        config.json, log.jsonl (per step), summary.json   (committed)
figures/           generated charts
data/ models/ checkpoints/   large, regenerable (gitignored)
```

## Limitations

The result holds for this setup only:

- One small model (0.6B) on one machine (M5, 32 GB).
- These library versions.
- Hand-written training loops. The stock trainers (`mlx_lm.lora --fine-tune-type full`, the HF Trainer)
  differ in batching, padding and logging overhead.
- PyTorch without `torch.compile`.

A larger model, a different batch size or newer versions could change the gap. Both frameworks have been
improving quickly.

## Data and model licenses

- Qwen3-0.6B-Base: Apache 2.0.
- arXiv papers: the Common Pile subset of papers whose authors chose CC BY, CC BY-SA or CC0; see the
  [dataset card](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered).
- wikitext-103: CC BY-SA.

None of the data or weights are committed to this repo.
