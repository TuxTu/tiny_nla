#!/bin/bash
set -euo pipefail
SGLANG_LOG="/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/sglang_rl_16789840.log"
CKPT_DIR="/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/checkpoints/rl"
SNAP_DIR="/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/checkpoints/rl_checkpoints"
declare -A MILESTONES=([101]=100 [201]=200 [251]=250)
last_count=0
while true; do
    cur_count=$(grep -c "POST /update_weights_from_disk" "$SGLANG_LOG" 2>/dev/null || echo 0)
    if [ "$cur_count" -gt "$last_count" ]; then
        last_count="$cur_count"
        target_step="${MILESTONES[$cur_count]:-}"
        if [ -n "$target_step" ]; then
            echo "[$(date)] Step $target_step (count=$cur_count). Snapshotting..."
            sleep 10
            mkdir -p "$SNAP_DIR/step_${target_step}/actor" "$SNAP_DIR/step_${target_step}/critic"
            cp -r "$CKPT_DIR/actor/"* "$SNAP_DIR/step_${target_step}/actor/" 2>/dev/null || true
            cp -r "$CKPT_DIR/critic/"* "$SNAP_DIR/step_${target_step}/critic/" 2>/dev/null || true
            echo "[$(date)] Snapshot step_${target_step} done."
        fi
    fi
    STATUS=$(squeue -u x_tuhan -h -o "%T" -j 16789840 2>/dev/null | head -1)
    [ -z "$STATUS" ] && { echo "[$(date)] Job finished."; break; }
    sleep 5
done
