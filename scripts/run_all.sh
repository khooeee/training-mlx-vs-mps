#!/bin/zsh
# Full comparison: training runs one at a time (never concurrent), alternating frameworks,
# then conversion, perplexity eval and the report. Meant to run under caffeinate:
#
#   caffeinate -is scripts/run_all.sh <fp32_micro_batch> <bf16_micro_batch> <total_steps>
#
# A failed step is logged and the rest continue. Progress: runs/logs/run_all.log

FP32_MB=$1 BF16_MB=$2 STEPS=$3
cd "${0:A:h}/.."
PY=.venv/bin/python
mkdir -p runs/logs
LOG=runs/logs/run_all.log
COOLDOWN=60  # seconds idle between runs so each starts from a similar thermal state

step() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $LOG
  "$@" > runs/logs/$name.out 2>&1
  echo "$(date '+%F %T') end   $name (exit $?)" >> $LOG
}

train() {
  step "$@"
  sleep $COOLDOWN
}

echo "$(date '+%F %T') run_all fp32_mb=$FP32_MB bf16_mb=$BF16_MB steps=$STEPS" >> $LOG
train mlx-fp32      env MLX_ENABLE_TF32=0 $PY scripts/train_mlx.py   --dtype float32  --micro-batch $FP32_MB --total-steps $STEPS --save
train torch-fp32                          $PY scripts/train_torch.py --dtype float32  --micro-batch $FP32_MB --total-steps $STEPS --save
train mlx-bf16                            $PY scripts/train_mlx.py   --dtype bfloat16 --micro-batch $BF16_MB --total-steps $STEPS --save
train torch-bf16                          $PY scripts/train_torch.py --dtype bfloat16 --micro-batch $BF16_MB --total-steps $STEPS --save
train mlx-fp32-tf32 env MLX_ENABLE_TF32=1 $PY scripts/train_mlx.py   --dtype float32  --micro-batch $FP32_MB --total-steps $STEPS --save --run-name mlx-fp32-tf32
step convert $PY scripts/convert.py
step eval    $PY scripts/eval_ppl.py
step report  $PY scripts/report.py
echo "$(date '+%F %T') run_all done" >> $LOG
