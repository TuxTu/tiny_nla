#!/bin/bash
# One-shot status for the actor retrain (job 17361668) + the loss-mask control
# (17361671). Safe to run any time, from any session -- reads only log files.
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
LOG=$PROJ/logs/actor_fix_17361668.out

echo "=== queue ==="
squeue -u x_tuhan -o "%.10i %.18j %.9T %.10M %.11l %R" 2>/dev/null | grep -E "JOBID|17361668" || echo "  (17361668 no longer queued -- it finished)"

echo
echo "=== retrain: last 6 evals (target: 1461 steps) ==="
grep -E "\[eval\]" "$LOG" | tail -6

echo
echo "=== retrain: FINAL ABLATION (the number that matters) ==="
if grep -q "shuffled - correct" "$LOG"; then
    sed -n '/ablation on HELD-OUT/,/VERDICT/p' "$LOG"
    echo
    echo "Reference points:"
    echo "  +0.0013  broken split50 run (unmasked loss)   -- ignoring the vector"
    echo "  +0.0054  control, unmasked arm, 300 steps     -- ignoring the vector"
    echo "  +0.1462  control, masked arm, 300 steps       -- WEAK"
    echo "  ~0.20    original Qwen2.5-7B healthy reference"
    echo "  >0.25    script's threshold for 'genuinely conditions'"
else
    echo "  not reached yet -- training still in progress"
fi

echo
echo "=== errors, if any ==="
grep -iE "Traceback|out of memory|CANCELLED|DUE TO TIME" "$PROJ/logs/actor_fix_17361668.err" 2>/dev/null | tail -5 || true
echo "  (none above = clean)"
