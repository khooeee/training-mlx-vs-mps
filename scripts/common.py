"""Shared config, data order, LR schedule, logging and system stats for both trainers.

Imports only numpy and the stdlib, so neither trainer loads the other framework. Works on
macOS (Apple Silicon) and Linux (NVIDIA); each machine's results go to runs/<machine>/.
"""

import argparse
import ctypes
import json
import math
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"  # machine-local and gitignored, so not namespaced
BASE_MODEL = ROOT / "models" / "Qwen3-0.6B-Base"

SEQ_LEN = 1024  # stored rows are SEQ_LEN + 1: inputs plus the shifted targets
WARMUP_EXCLUDE = 20  # steps excluded from steady-state throughput (compile, shader caching)

# Every run the scripts know about, in report order.
ALL_RUNS = ["mlx-fp32", "torch-fp32", "mlx-bf16", "torch-bf16", "mlx-fp32-tf32", "torch-fp32-tf32"]

IS_MAC = platform.system() == "Darwin"


@dataclass
class TrainConfig:
    dtype: str = "float32"  # "float32" or "bfloat16"; weights, grads and optimizer state all use it
    tf32: bool = False  # fp32 matmuls in TF32 (MLX: MLX_ENABLE_TF32=1; PyTorch: CUDA only)
    micro_batch: int = 1
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
        flag = f"--{name.replace('_', '-')}"
        if isinstance(default, bool):
            p.add_argument(flag, action="store_true")
        else:
            p.add_argument(flag, type=type(default), default=default)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (probes); schedule still uses --total-steps")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save", action="store_true", help="save the trained model to checkpoints/<run-name>")
    args = p.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    tag = "fp32" if cfg.dtype == "float32" else "bf16"
    args.run_name = args.run_name or f"{framework}-{tag}{'-tf32' if cfg.tf32 else ''}"
    return args, cfg


def load_split(name):
    return np.load(DATA / f"{name}.npy", mmap_mode="r")


class Batches:
    """Seeded permutation of training rows, consumed in order with no repeats.

    Each optimizer step uses the same rows whatever the micro-batch size, so losses
    stay comparable across micro-batch settings, frameworks and machines.
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


# --- machine identity and system readouts -------------------------------------------------


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia(query):
    """Fields of `nvidia-smi --query-gpu` for the first GPU, or [] without an NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    return [v.strip() for v in out.splitlines()[0].split(",")] if out else []


def _meminfo_kb(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return 0


def hardware():
    if IS_MAC:
        return {
            "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_gb": int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30,
        }
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    info = {"cpu": cpu, "memory_gb": _meminfo_kb("MemTotal") / 2**20}
    gpu = _nvidia("name,memory.total,driver_version,power.limit")
    if gpu:
        info.update(gpu=gpu[0], gpu_memory_gb=float(gpu[1]) / 1024, nvidia_driver=gpu[2], gpu_power_limit_w=gpu[3])
    return info


def machine_tag():
    """Directory name for this machine's results, e.g. apple-m5-32gb. Override with BENCH_MACHINE."""
    if os.environ.get("BENCH_MACHINE"):
        return os.environ["BENCH_MACHINE"]
    hw = hardware()
    if "gpu" in hw:
        name, mem = hw["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", ""), hw["gpu_memory_gb"]
    else:
        name, mem = hw.get("chip") or platform.machine(), hw["memory_gb"]
    return re.sub(r"[^a-z0-9]+", "-", f"{name} {round(mem)}gb".lower()).strip("-")


MACHINE = machine_tag()
RUNS = ROOT / "runs" / MACHINE


def system_info():
    return {"machine": MACHINE, "os": platform.platform(), "python": platform.python_version(), **hardware()}


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib") if IS_MAC else None


def _mac_footprint():
    buf = (ctypes.c_uint64 * 64)()
    # RUSAGE_INFO_V4: a 16-byte uuid, then uint64 fields; phys_footprint is field 7,
    # lifetime_max_phys_footprint field 28.
    if _libc.proc_pid_rusage(os.getpid(), 4, buf) != 0:
        return None, None
    return buf[2 + 7], buf[2 + 28]


def _nvidia_process_mem():
    out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        pid, mib = (v.strip() for v in line.split(","))
        if pid == str(os.getpid()):
            return int(mib) * 2**20
    return None


class MemoryMeter:
    """The one memory number that means the same thing for every framework on a machine.

    macOS: the process's physical footprint as the OS accounts it (CPU and unified GPU memory
    alike), with the OS's own lifetime peak. Linux/NVIDIA: GPU memory nvidia-smi attributes to
    this process, sampled after each step (running max, so a slight underestimate of the peak).
    """

    def __init__(self):
        self.peak = 0

    def sample(self):
        cur, peak = _mac_footprint() if IS_MAC else (_nvidia_process_mem(), None)
        self.peak = max(self.peak, peak or 0, cur or 0)
        return cur, self.peak


def swap_used_mb():
    if IS_MAC:
        out = _run(["sysctl", "-n", "vm.swapusage"])
        return float(out.split("used =")[1].split("M")[0])
    return (_meminfo_kb("SwapTotal") - _meminfo_kb("SwapFree")) / 1024


def thermal_state():
    if IS_MAC:
        return f"thermal pressure {thermal_level()} (0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping)"
    gpu = _nvidia("temperature.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit")
    return f"gpu {gpu[0]}C, sm clock {gpu[1]}/{gpu[2]} MHz, power {gpu[3]}/{gpu[4]} W" if gpu else ""


def is_cool():
    """True once the machine is back to a baseline thermal state (run_all.sh waits for this between runs)."""
    level = thermal_level()
    return level is None or (level == 0 if IS_MAC else level < 50)


def thermal_level():
    """Cheap per-step throttling indicator.

    macOS: system thermal pressure level (0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping); a
    fanless Mac reaches 2 within minutes of sustained GPU load and slows down. Linux/NVIDIA: GPU
    temperature in °C.
    """
    if IS_MAC:
        out = _run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"]).split()
        return int(out[-1]) if out and out[-1].isdigit() else None
    gpu = _nvidia("temperature.gpu")
    return int(gpu[0]) if gpu else None


# --- logging -----------------------------------------------------------------------------


class RunLog:
    def __init__(self, name, config):
        self.dir = RUNS / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self.f = open(self.dir / "log.jsonl", "w")

    def log(self, **rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


def summarize(records):
    """Throughput and memory from per-step log records.

    tokens_per_s: median step after the first WARMUP_EXCLUDE steps (compile, shader caching).
    tokens_per_s_sustained: median step over the second half of the run, when a fanless machine
    has reached its throttled steady state. That is the number to compare across runs of
    different lengths on a machine that throttles.
    """
    steady = [r for r in records if r["step"] >= WARMUP_EXCLUDE] or records
    late = records[len(records) // 2 :]
    times = np.array([r["step_time"] for r in steady])
    toks = steady[0]["tokens"]
    thermal = [r.get("thermal") for r in records if r.get("thermal") is not None]
    return {
        "steps": len(records),
        "steady_steps": len(steady),
        "first_step_s": records[0]["step_time"],
        "step_time_median_s": float(np.median(times)),
        "step_time_p90_s": float(np.percentile(times, 90)),
        "tokens_per_s": toks / float(np.median(times)),
        "tokens_per_s_sustained": toks / float(np.median([r["step_time"] for r in late])),
        "wall_time_s": float(sum(r["step_time"] for r in records)),
        "thermal_levels": {str(v): thermal.count(v) for v in sorted(set(thermal))},
        "peak_footprint_gb": max(r["peak_footprint"] or 0 for r in records) / 1e9,
        "final_loss": records[-1]["loss"],
    }
