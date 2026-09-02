#!/bin/bash
#SBATCH --job-name=nla-collect
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/collect_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/collect_%j.err
# Gathers every result from the split50 pipeline into ONE readable file.
set -uo pipefail
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
cd "$PROJ"
R="$PROJ/RESULTS_split50.md"

{
echo "# NLA split50 pipeline results"
echo ""
echo "Generated $(date '+%Y-%m-%d %H:%M')"
echo ""
echo "Doc-disjoint quartering: AV 125k labeled / AR 125k labeled / RL 250k unlabeled."
echo "Eval set = ar pool, 914 documents / 9,042 rows, disjoint from everything."
echo ""
echo "## Job outcomes"
echo '```'
sacct -j "${CRITIC_JOB},${ACTOR_JOB},${E2E_JOB},${RL_JOB}" \
      --format=JobID%14,JobName%18,State%14,Elapsed 2>/dev/null | grep -v '\.ba\|\.ex\|\.0'
echo '```'
echo ""
echo "## Critic (AR-SFT) — held-out FVE trajectory"
echo '```'
grep -ah "eval @ step\|best held-out" logs/critic_s50_${CRITIC_JOB}.out 2>/dev/null || echo "(no critic log)"
echo '```'
echo ""
echo "For reference, the previous 249k-row critic that shared documents with the actor:"
echo "  +55.4 @100 -> +62.3 @300 -> +63.7 @500 -> +65.19 @974 (final), 65.72 on all 9042 rows"
echo ""
echo "## Actor (AV-SFT) — held-out cross-entropy"
echo '```'
grep -ah "\[eval\]" logs/actor_s50_${ACTOR_JOB}.out 2>/dev/null || echo "(no actor log)"
echo '```'
echo ""
echo "## End-to-end FVE  (vector -> actor explanation -> critic -> vector)"
echo '```'
grep -ahA3 "END-TO-END\|CRITIC-ONLY" logs/e2e_eval_${E2E_JOB}.out 2>/dev/null
grep -ah "Samples evaluated\|^MSE:\|baseline (\|FVE_nrm\|subset-matched" logs/e2e_eval_${E2E_JOB}.out 2>/dev/null || echo "(no e2e log)"
echo '```'
echo ""
echo "## RL (GRPO) smoke — 10 steps"
echo '```'
grep -ah "^\[rl\] step=" logs/rl_smoke_v2_${RL_JOB}.out 2>/dev/null || echo "(no rl log)"
echo '```'
echo ""
echo "extract_fail is the count of rollouts with no parseable <explanation> tag."
echo "Those receive FAILED_EXTRACTION_REWARD = -2.0. If that number is large,"
echo "GRPO is optimising tag syntax rather than semantics and the SFT needs work"
echo "before a full RL run is worth launching."
echo ""
echo "## Failures seen"
echo '```'
grep -ah "Traceback\|CUDA out of memory\|AssertionError\|DUE to\|SIGABRT" \
     logs/critic_s50_${CRITIC_JOB}.err logs/actor_s50_${ACTOR_JOB}.err \
     logs/e2e_eval_${E2E_JOB}.err logs/rl_smoke_v2_${RL_JOB}.err 2>/dev/null | sort -u | head -20 \
     || echo "(none)"
echo '```'
} > "$R"

echo "wrote $R"
cat "$R"
