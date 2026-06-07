#!/bin/bash
# Snapshot RL checkpoints at save_every milestones (100, 200).
# Monitors SGLang's update_weights_from_disk count.
set -euo pipefail

SGLANG_LOG="/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/sglang_rl_16789804.log"
CKPT_DIR="/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/checkpoints/rl"
SNAPSHOT_DIR="/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/checkpoints/rl_checkpoints"

# Count map: SGLang update_weights count -> step that just completed
# step 0 -> count 1, step 1 -> count 2, ..., step N -> count N+1
# So count=101 means step 100's per-step save just pushed.
# _save() runs after that, then step 101 starts.

last_count=0
declare -A MILESTONES=([101]=100 [201]=200 [251]=250)

while true; do
    cur_count=$(grep -c "POST /update_weights_from_disk" "$SGLANG_LOG" 2>/dev/null || echo 0)
    
    if [ "$cur_count" -gt "$last_count" ]; then
        last_count="$cur_count"
        
        # Check if this count matches a milestone
        target_step="${MILESTONES[$cur_count]:-}"
        if [ -n "$target_step" ]; then
            echo "[$(date)] Step $target_step detected (count=$cur_count). Waiting for _save() to finish..."
            # _save() runs after the per-step save + SGLang update.
            # Wait for critic model to be saved (indicates _save() completed).
            sleep 5
            # Wait up to 60s for critic model to appear (saving can take ~13s)
            for i in $(seq 1 12); do
                if [ -f "$CKPT_DIR/critic/model.safetensors" ]; then
                    ctime=$(stat -c %Y "$CKPT_DIR/critic/model.safetensors")
                    now=$(date +%s)
                    if [ $((now - ctime)) -lt 30 ]; then
                        echo "[$(date)] critic model appears fresh — _save() done."
                        break
                    fi
                fi
                sleep 5
            done
            
            mkdir -p "$SNAPSHOT_DIR/step_${target_step}/actor" "$SNAPSHOT_DIR/step_${target_step}/critic"
            echo "[$(date)] Snapshotting step $target_step..."
            cp -r "$CKPT_DIR/actor/"* "$SNAPSHOT_DIR/step_${target_step}/actor/" 2>/dev/null || true
            cp -r "$CKPT_DIR/critic/"* "$SNAPSHOT_DIR/step_${target_step}/critic/" 2>/dev/null || true
            echo "[$(date)] Snapshot for step $target_step saved to $SNAPSHOT_DIR/step_${target_step}/"
            
            # Verify
            if [ -f "$SNAPSHOT_DIR/step_${target_step}/actor/model.safetensors" ]; then
                echo "[$(date)] ✓ Actor checkpoint verified."
            fi
            if [ -f "$SNAPSHOT_DIR/step_${target_step}/critic/model.safetensors" ]; then
                echo "[$(date)] ✓ Critic checkpoint verified."
            fi
        fi
    fi
    
    # Check if job is still running
    STATUS=$(squeue -u x_tuhan -h -o "%T" 2>/dev/null | head -1)
    if [ -z "$STATUS" ]; then
        echo "[$(date)] Job finished. Final count=$cur_count."
        break
    fi
    
    sleep 5
done
echo "[$(date)] Snapshot monitor exiting."
