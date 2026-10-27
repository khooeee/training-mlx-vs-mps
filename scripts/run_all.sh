#!/usr/bin/env bash
# Full comparison on this machine: training runs one at a time (never concurrent), alternating
# frameworks, then conversion, perplexity eval and the report.
#
#   scripts/run_all.sh <fp32_micro_batch> <bf16_micro_batch> <total_steps>
#
# macOS: wrap with `caffeinate -is` so the machine doesn't sleep. Linux: keep it on AC power and
# awake (e.g. `systemd-inhibit --what=sleep:idle scripts/run_all.sh ...`).
#
# FRAMEWORKS defaults to "mlx torch", or "torch" if MLX isn't installed. TF32 variants run for MLX,
# and for PyTorch only on CUDA (PyTorch has no TF32 mode on MPS). A failed step is logged and the
# rest continue. Progress: runs/<machine>/logs/run_all.log

set -u
FP32_MB=$1 BF16_MB=$2 STEPS=$3
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MACHINE=$($PY -c "import sys; sys.path.insert(0, 'scripts'); import common; print(common.MACHINE)")
LOGDIR=runs/$MACHINE/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/run_all.log
COOLDOWN=60  # seconds idle between runs so each starts from a similar thermal state

if [ -z "${FRAMEWORKS:-}" ]; then
  if $PY -c "import mlx.core" 2>/dev/null; then FRAMEWORKS="mlx torch"; else FRAMEWORKS="torch"; fi
fi
HAS_CUDA=$($PY -c "import torch; print(int(torch.cuda.is_available()))")

step() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> "$LOG"
  "$@" > "$LOGDIR/$name.out" 2>&1
  echo "$(date '+%F %T') end   $name (exit $?)" >> "$LOG"
}

train() {  # train <framework> <dtype> <micro_batch> [extra args...]
  local fw=$1 dtype=$2 mb=$3; shift 3
  local name="$fw-$([ "$dtype" = float32 ] && echo fp32 || echo bf16)"
  [[ " $* " == *" --tf32 "* ]] && name="$name-tf32"
  step "$name" $PY "scripts/train_$fw.py" --dtype "$dtype" --micro-batch "$mb" --total-steps "$STEPS" --save --run-name "$name" "$@"
  sleep $COOLDOWN
}

echo "$(date '+%F %T') run_all machine=$MACHINE frameworks='$FRAMEWORKS' fp32_mb=$FP32_MB bf16_mb=$BF16_MB steps=$STEPS" >> "$LOG"
for fw in $FRAMEWORKS; do train "$fw" float32 "$FP32_MB"; done
for fw in $FRAMEWORKS; do train "$fw" bfloat16 "$BF16_MB"; done
for fw in $FRAMEWORKS; do
  if [ "$fw" = mlx ] || [ "$HAS_CUDA" = 1 ]; then train "$fw" float32 "$FP32_MB" --tf32; fi
done
if [[ " $FRAMEWORKS " == *" mlx "* ]]; then
  step convert $PY scripts/convert.py
  step eval    $PY scripts/eval_ppl.py
fi
step report $PY scripts/report.py
echo "$(date '+%F %T') run_all done" >> "$LOG"
