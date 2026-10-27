"""Shared config, data order, LR schedule, logging and system stats for both trainers.

Imports only numpy and the stdlib, so neither trainer loads the other framework.
"""

import argparse
import ctypes
import json
import math
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
CKPT = ROOT / "checkpoints"
BASE_MODEL = ROOT / "models" / "Qwen3-0.6B-Base"

SEQ_LEN = 1024  # stored rows are SEQ_LEN + 1: inputs plus the shifted targets
WARMUP_EXCLUDE = 20  # steps excluded from steady-state throughput (compile, shader caching)


@dataclass
class TrainConfig:
    dtype: str = "float32"  # "float32" or "bfloat16"; weights, grads and optimizer state all use it
    micro_batch: int = 4
    tokens_per_step: int = 16384
    total_steps: int = 200  # length of the LR schedule
    peak_lr: float = 2e-5
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1  # applied to every parameter in both frameworks
    max_grad_norm: float = 1.0
    seed: int = 0

    @property
    def grad_accum(self):
        seqs = self.tokens_per_step // SEQ_LEN
        assert seqs % self.micro_batch == 0, "micro_batch must divide sequences per step"
        return seqs // self.micro_batch

    @property
    def warmup_steps(self):
        return max(1, round(self.warmup_frac * self.total_steps))


def lr_at(cfg, step):
    """Linear warmup then cosine decay to min_lr_frac * peak. `step` is the 0-based optimizer step."""
    if step < cfg.warmup_steps:
        return cfg.peak_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    min_lr = cfg.peak_lr * cfg.min_lr_frac
    return min_lr + (cfg.peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def parse_args(framework):
    p = argparse.ArgumentParser()
    for name, default in asdict(TrainConfig()).items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (probes); schedule still uses --total-steps")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save", action="store_true", help="save the trained model to checkpoints/<run-name>")
    args = p.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    args.run_name = args.run_name or f"{framework}-{'fp32' if cfg.dtype == 'float32' else 'bf16'}"
    return args, cfg


def load_split(name):
    return np.load(DATA / f"{name}.npy", mmap_mode="r")


class Batches:
    """Seeded permutation of training rows, consumed in order with no repeats.

    Each optimizer step uses the same rows whatever the micro-batch size, so losses
    stay comparable across micro-batch settings as well as across frameworks.
    """

    def __init__(self, cfg, n_steps):
        self.data = load_split("train")
        self.mb, self.accum = cfg.micro_batch, cfg.grad_accum
        need = n_steps * self.accum * self.mb
        assert need <= len(self.data), f"need {need} rows, have {len(self.data)}"
        self.order = np.random.default_rng(cfg.seed).permutation(len(self.data))[:need]

    def get(self, step, micro):
        i = (step * self.accum + micro) * self.mb
        rows = np.asarray(self.data[self.order[i : i + self.mb]])
        return rows[:, :-1].astype(np.int32), rows[:, 1:].astype(np.int32)


class RunLog:
    def __init__(self, name, config):
        self.dir = RUNS / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self.f = open(self.dir / "log.jsonl", "w")

    def log(self, **rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def footprint():
    """(current, lifetime peak) physical footprint of this process in bytes, as macOS accounts it.

    Counts CPU and unified GPU allocations alike, so it's the one memory number that
    means the same thing for both frameworks.
    """
    buf = (ctypes.c_uint64 * 64)()
    # RUSAGE_INFO_V4: a 16-byte uuid, then uint64 fields; phys_footprint is field 7,
    # lifetime_max_phys_footprint field 28.
    if _libc.proc_pid_rusage(os.getpid(), 4, buf) != 0:
        return None, None
    return buf[2 + 7], buf[2 + 28]


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used =")[1].split("M")[0])


def thermal_state():
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return " | ".join(line.strip() for line in out.splitlines() if line.strip())


def system_info():
    return {"macos": platform.mac_ver()[0], "python": platform.python_version(), "machine": platform.machine()}


def summarize(records):
    """Steady-state throughput and memory from per-step log records."""
    steady = [r for r in records if r["step"] >= WARMUP_EXCLUDE] or records
    times = np.array([r["step_time"] for r in steady])
    toks = steady[0]["tokens"]
    return {
        "steps": len(records),
        "steady_steps": len(steady),
        "first_step_s": records[0]["step_time"],
        "step_time_median_s": float(np.median(times)),
        "step_time_p90_s": float(np.percentile(times, 90)),
        "tokens_per_s": toks / float(np.median(times)),
        "peak_footprint_gb": max(r["peak_footprint"] for r in records) / 1e9,
        "final_loss": records[-1]["loss"],
    }
