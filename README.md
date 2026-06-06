# tiny_nla

Natural Language Autoencoders pipeline for small HuggingFace models (< 4B).

Decoupled labeling → reusable across any model sharing a Qwen3 tokenizer.
Training pipeline: critic SL → actor SFT → GRPO RL, using only HF Transformers + PyTorch.

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

## Quick Start

```bash
pip install -e .

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

# 5. RL training (GRPO)
python -m nla.training.train_rl \
  --data data/qwen3_0.6b/rl_train.parquet \
  --model-name Qwen/Qwen3-0.6B \
  --actor-ckpt data/checkpoints/actor_sft \
  --critic-ckpt data/checkpoints/critic_sft \
  --output-dir data/checkpoints/rl --n-samples 8 --num-steps 200
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

## License

Apache-2.0
