# tiny_nla

Natural Language Autoencoders pipeline for small HuggingFace models (< 4B).

Decoupled labeling → reusable across any model sharing a Qwen3 tokenizer.
Training pipeline: critic SL → actor SFT → GRPO RL, using only HF Transformers + PyTorch.

## Design

This is a from-scratch reimplementation of the NLA pipeline, not a fork of Anthropic's
original [Miles](https://github.com/anthropics/natural_language_autoencoders) framework.
Two design choices drive this:

**1. Data parallelism instead of model parallelism.**
Miles targets models large enough to require FSDP or Megatron-style sharding across
multiple GPUs. tiny_nla targets models that *fit on a single GPU* (≤ 4B parameters).
Instead of sharding one giant model, we run one copy per GPU and use PyTorch DDP
with a DistributedSampler — each rank sees unique data, gradients are averaged.
This gives linear throughput scaling up to the GPU count with standard PyTorch
primitives and no framework lock-in.

**2. Decoupled labeling keeps training cheap.**
API-labeled explanations are tied to a *tokenizer*, not a specific model. All Qwen3
variants (0.6B–8B) share the same tokenizer, so the same 250k labeled positions
work for the entire family. You only pay for labeling once, then extract vectors
from whatever model you want to train. This cuts the dominant cost (API calls) by
an order of magnitude compared to per-model labeling.

Together these choices make NLA training practical on a single 8-GPU node with
commodity hardware — no InfiniBand, no Ray cluster, no Megatron — while still
producing models that transfer across the Qwen3 family.

## Architecture

```
nla/
  datagen/                    Data generation pipeline
    run_pipeline.py           Orchestrator: extract → explain → output → vectors
    extract_positions.py      Tokenize corpus, sample positions, decode text (CPU)
    api_explain.py            Label positions via DeepSeek/Anthropic API
    extract_vectors.py        Extract hidden states at positions (GPU/MPS)
    split_positions.py        Doc-level three-way split into AV/AR/RL buckets
    providers.py              DeepSeek + Anthropic completion backends
    _common.py                YAML config loading utilities
  training/                   Training pipeline
    build_training_data.py    Join explained + vectors → training-ready parquets
    train_critic_sft.py       AR-SFT: train truncated model for text→vector (MSE)
    train_actor_sft.py        AV-SFT: train full model for vector→text (CE + injection)
    train_rl.py               GRPO RL: joint actor+critic training
    models.py                 NLACriticModel (truncated transformer + value head)
    injection.py              inject_at_marked_positions() forward hook
    loss.py                   nla_critic_loss, sft_loss
    schema.py                 Shared constants, normalization, token metadata
    injection_tokens.py       Auto-discover injection characters, compute neighbors
    sidecar.py                nla_meta.yaml read/write
    env_config.py             Hardware auto-detection (CUDA/MPS/CPU)
configs/
  datagen_0.6b_25k.yaml      Datagen: 25k docs, 0.6B
  datagen_smoke_100.yaml      Datagen: 100-doc quick test
  train_0.6b.yaml             Training: critic + actor + RL
```

## Pre-labeled Dataset

[`TuHan/qwen3-nla-250k`](https://huggingface.co/datasets/TuHan/qwen3-nla-250k) is a pre-labeled NLA dataset for the Qwen3 tokenizer family (0.6B, 1.7B, 4B, 8B). It contains 250k FineWeb text snippets (~499k labeled rows), each annotated by **DeepSeek V4 Flash** with a structured explanation of what semantic/structural signal a language model's activation vector encodes at that position.

### Dataset structure

```
Dataset: TuHan/qwen3-nla-250k  (499k rows)
├── doc_id                        FineWeb document identifier
├── n_raw_tokens                  Token count in the context window
├── detokenized_text_truncated    Text snippet for AV extraction
└── api_explanation               Structured explanation (DeepSeek V4 Flash)
```

Each `api_explanation` describes 2-3 semantic features of the activation vector — syntactic constraints, topic continuation, register shifts, or entity tracking — in free-form natural language. The text positions are sampled across 50k FineWeb documents (5 positions per doc), producing ~499k labeled rows deterministically split:

| Split | Rows | Purpose |
|-------|------|---------|
| AV-SFT | 125k | Train actor: vector → explanation |
| AR-SFT | 125k | Train critic: explanation → vector |
| RL | 250k | GRPO fine-tuning (on-policy) |

### Using the dataset

```python
from datasets import load_dataset

ds = load_dataset("TuHan/qwen3-nla-250k", split="train")
# ds[0]:
# {
#   "doc_id": "HuggingFaceFW/fineweb:train:2",
#   "n_raw_tokens": 210,
#   "detokenized_text_truncated": "A novel two-step immunotherapy approach...",
#   "api_explanation": "Syntactic/structural constraints: the conjunction..."
# }
```

### Compatibility

The dataset is **tokenizer-bound** — labels embed token IDs from Qwen3's vocabulary. All Qwen3 variants (0.6B-8B) share the same tokenizer, so the labels work across the entire model family. A SHA-256 tokenizer fingerprint is embedded in every output parquet for downstream verification.

To extract activation vectors from your own model:

```bash
# Extract hidden states at labeled positions (works with any Qwen3 model)
python -m nla.datagen.extract_vectors \
    --explained data/av_sft_explained.parquet \
    --model Qwen/Qwen3-4B --output data/av_sft_vectors.parquet
```

Then build training-ready parquets with `build_training_data.py` (see Quick Start below).

## Quick Start

```bash
pip install -e .            # core pipeline
pip install -e ".[rl]"      # + SGLang for fast RL rollout (requires GPU)
```

```bash
# 1. Generate labels + vectors
python -m nla.datagen.run_pipeline --config configs/datagen_0.6b_25k.yaml

# 2. Build training parquets (join explanations + vectors)
for split in av_sft ar_sft rl; do
  python -m nla.training.build_training_data \
    --explained data/qwen3_0.6b/${split}_explained.parquet \
    --vectors data/qwen3_0.6b/${split}_explained_vectors.parquet \
    --tokenizer Qwen/Qwen3-0.6B \
    --output data/qwen3_0.6b/${split}_train.parquet --split-type $split
done

# 3. Train critic (AR: text → vector)
python -m nla.training.train_critic_sft \
  --data data/qwen3_0.6b/ar_sft_train.parquet \
  --model-name Qwen/Qwen3-0.6B \
  --output-dir data/checkpoints/critic_sft --num-steps 1000

# 4. Train actor (AV: vector → text)
python -m nla.training.train_actor_sft \
  --data data/qwen3_0.6b/av_sft_train.parquet \
  --model-name Qwen/Qwen3-0.6B \
  --output-dir data/checkpoints/actor_sft --num-steps 1000

# 5. RL training (GRPO) — default: HF generate()
python -m nla.training.train_rl \
  --data data/qwen3_0.6b/rl_train.parquet \
  --model-name Qwen/Qwen3-0.6B \
  --actor-ckpt data/checkpoints/actor_sft \
  --critic-ckpt data/checkpoints/critic_sft \
  --output-dir data/checkpoints/rl --n-samples 8 --num-steps 200

# 5b. RL with SGLang (faster — requires pip install -e ".[rl]")
python -m nla.training.train_rl \
  --data data/qwen3_0.6b/rl_train.parquet \
  --actor-ckpt data/checkpoints/actor_sft \
  --critic-ckpt data/checkpoints/critic_sft \
  --output-dir data/checkpoints/rl --n-samples 8 --num-steps 200 \
  --use-sglang --sglang-mem-fraction 0.7
```

## Pipeline stages

### Data generation (`nla.datagen.run_pipeline`)

| Stage | What it does | Output |
|-------|-------------|--------|
| `extract` | Tokenize corpus, sample positions, decode text | `pool/{av,ar,rl}/positions.parquet` |
| `explain` | Label via DeepSeek API (AV and AR only) | `pool/{av,ar}/explained.parquet` |
| `output` | Deterministic subsample from pool | `output/{av,ar,rl}_*.parquet` |
| `vectors` | GPU forward pass, extract hidden states | `output/*_vectors.parquet` |

The pool grows monotonically — labels are never wasted. Models sharing a tokenizer reuse the same labels.

### Training (`nla.training`)

| Stage | Direction | Model | Loss |
|-------|----------|-------|------|
| Critic SL | text → vector | Truncated (K+1 layers + value head) | MSE (normalized) |
| Actor SFT | vector → text | Full model + injection hook | CE (response tokens only) |
| RL (GRPO) | both | Actor + critic, on-policy | Policy gradient + KL + MSE |

## Configuration

All stages configured via YAML:

```yaml
tokenizer_name: Qwen/Qwen3-0.6B
model_name: Qwen/Qwen3-0.6B
corpus: {name: HuggingFaceFW/fineweb, config: sample-10BT, split: train}
positions_per_doc: 10
min_position: 50
max_length: 2048
seed: 42
pool_dir: data/pool
output_dir: data/output
num_docs: 25000
split: {av_sft: 0.25, ar_sft: 0.25, rl: 0.50}
batch_size: 2
provider: {name: deepseek}
```

## Tokenizer fingerprinting

Every parquet embeds a SHA-256 fingerprint of the tokenizer (sorted vocab, special tokens, BOS/EOS/PAD IDs). Downstream consumers verify compatibility before loading — a mismatch means labels don't correspond to the same token positions. All Qwen3 variants share the same fingerprint.

## Providers

| Provider | Default model | Cost (/MTok) | Env var |
|----------|-------------|-------------|---------|
| DeepSeek | `deepseek-v4-flash` | $0.14 input / $0.28 output | `DEEPSEEK_API_KEY` |
| Anthropic | `claude-haiku-4-5-20251001` | $0.80 input / $4.00 output | `ANTHROPIC_API_KEY` |

DeepSeek v4-flash is ~11× cheaper and outperforms Haiku 4.5 on SWE-bench (79.0 vs 73.3).

## Hardware

Qwen3-0.6B fits on any single GPU (Colab T4/L4/A100, Apple Silicon). Data generation is CPU-only.
For Qwen3-4B RL training, we used 8× A100 80GB (GPU0=SGLang, GPU1-7=DDP training).

## Model & training config (Qwen3-4B)

| Setting | Value |
|---------|-------|
| Base model | Qwen3-4B (36 layers, d_model=2560, GQA 32Q/8KV) |
| Critic | 25 layers (2/3 truncated) + value_head, identity-init |
| Injection | char `㈎`, scale `2.5 × √d_model = 126.49` |
| Extraction layer | 24 (2/3 × 36) |
| SFT (actor) | 250 steps, global_batch=256, 4×A100 DDP, CE loss |
| SFT (critic) | 250 steps, global_batch=256, 4×A100 DDP, MSE loss |
| RL (GRPO) | rollout_batch=2, n_samples=8, 7×A100 DDP + 1×A100 SGLang |
| Optimizer | ZeroRedundancyOptimizer (AdamW, lr=1.41e-5 constant) |
| max-response-len | 256 tokens |

## SFT baseline (Qwen3-4B)

| Metric | Actor SFT | Critic SFT |
|--------|----------|-----------|
| Training data | 125k rows | 125k rows |
| Steps | 250 | 250 |
| Final loss | CE 19.78 | MSE 1.06 |
| Extraction rate (greedy) | 100% | — |
| **FVE_nrm** | — | **−0.32** |
| **FVE_nrm_meannorm** | — | **−0.08** |

The critic FVE is negative — it predicts vectors *worse* than a constant
mean predictor.  To isolate whether the actor or critic is at fault, we
evaluated the critic directly on the API-labeled explanations (skipping
the actor entirely):

| Evaluation path | FVE_nrm | MSE |
|----------------|---------|-----|
| Full pipeline (actor → critic) | −0.32 | 0.774 |
| **Critic only** (API labels → critic) | **−0.22** | 0.680 |

Even with perfect API-labeled explanations, the critic cannot beat the
mean predictor.  The 4B critic (25/36 layers) is too small for
text→vector prediction.  This is a capacity limitation, not an actor
or training bug.

## RL Experiments — Bitter Lesson

We ran 7 GRPO RL attempts on Qwen3-4B with different KL configurations.
All used the same SFT checkpoints, 250k-row dataset, and 8× A100 GPUs.
**Every run mode-collapsed within 10–125 steps.**

| Run | KL type | kl_coef | Freeze critic? | Steps survived |
|-----|---------|---------|---------------|---------------|
| 1 | per_seq (signed) | 0.01 | No | ~20 (then fallbacks) |
| 2 | per_seq (signed) | 0.01 | No | ~125 (collapsed at eval) |
| 3 | per_seq (squared) | 0.01 | No | ~10 |
| 4 | per_seq (squared) | 0.1 | No | ~10 |
| 5 | per_seq (squared) | 1.0 | No | ~10 |
| 6 | per_seq (squared) | 0.1 | **Yes** | ~25 (collapsed at eval) |
| 7 | per_token (squared) | 0.1 | No | ~10 |

During training (SGLang path), rewards improved from −1.1 to −0.8 with
100% extraction.  At evaluation (HF generate), every checkpoint produced
degenerate output: `!!!!!!`, Chinese repetition, bilingual fragments.

The root cause: SGLang rollout (`input_embeds`) and HF generate
(`input_ids` + hook) produce different output distributions.  On a 4B
model, RL learns SGLang-specific patterns that don't transfer.  Combined
with a critic that can't beat the mean predictor (FVE < 0), the GRPO
signal is too noisy to improve explanation quality.

### What works

- **SFT training:** Actor (CE 19.78) and critic (MSE 1.06) train stably.
  The SFT actor produces coherent `<explanation>` tags with greedy decoding.
- **Data pipeline:** 250k labeled positions, reusable across Qwen3 family.
- **RL infrastructure:** SGLang rollout, GRPO, ZeroRedundancyOptimizer,
  DDP with DistributedSampler, checkpoint snapshots — all working.

## License

Apache-2.0
