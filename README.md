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

## Qwen3-8B results

Scaling the pipeline to Qwen3-8B (36 layers, d_model=4096, extraction layer 24)
on 100k UltraFineWeb documents. Both models train on **document-disjoint** halves
of the labeled pool, so no document seen by the critic is seen by the actor.

| Setting | Value |
|---------|-------|
| Base model | Qwen3-8B (36 layers, d_model=4096) |
| Critic | 25 layers (2/3 truncated) + value_head, identity-init |
| Injection | char `㈎`, scale 300 (mean vector L2 = 266) |
| AV pool | 124,741 pairs / 12,500 docs (10 positions per doc) |
| Actor SFT | 1,461 steps (3 epochs), global_batch=256, 4×A100 FSDP, fp32 master weights |
| Critic SFT | 487 steps (1 epoch), global_batch=256, 4×A100 |

| Metric | Actor (AV) | Critic (AR) |
|--------|-----------|------------|
| Final held-out CE | 1.3305 | — |
| Final held-out MSE | — | 0.2405 |
| **FVE_nrm_meannorm** | — | **+62.93%** |
| **FVE_nrm** | — | **+55.75%** |
| **Real-vs-shuffled gap** | **+0.4593** | — |

Both directions work at 8B. The critic beats the mean predictor by a wide margin
(the 4B critic could not, see above), and the actor genuinely conditions on the
injected vector.

### The loss mask is not optional

The actor initially appeared completely broken — a real-vs-shuffled gap of
**+0.0013**, i.e. swapping in a *different* vector cost the model nothing. Three
plausible-sounding explanations were wrong: data volume (a 1k→8k→64k→125k ladder
showed the gap still climbing at full data), positions-per-doc (Anthropic's
`qwen7b_ultrafineweb_100k.yaml` uses 10, the same as ours), and injection scale
(a 300-vs-160 A/B came back null — but that A/B ran for only 150 steps, too few
for the vector channel to exist at all, so it could not have detected a scale
effect; see *Injection scale* below for the corrected measurement).

The actual cause was `sft_loss()` being called **without a loss mask**, averaging
cross-entropy over prompt + response + padding instead of the response alone.
Padding was not even marked `-100`, so `pad == eos` was scored as a real target.
This halved the reported CE, which made the loss curve look converged when it was
not — so every run was also stopped far too early.

Both fixes were required; neither alone sufficed:

| loss | steps | real-vs-shuffled gap |
|------|-------|---------------------|
| unmasked | 487 | +0.0013 |
| unmasked | 300 | +0.0054 |
| masked | 150 | +0.0027 |
| masked | 300 | +0.1462 |
| **masked** | **1,461** | **+0.4593** |

The two arms at 300 steps differ *only* in the loss mask (`--legacy-unmasked-loss`
reproduces the old objective on demand) — a 27× difference from one line.

The injection-scale A/B deserves a caveat: it was run at 150 steps, which we now
know is too few for the vector channel to exist at all, so it could not have
detected a scale effect either way. Measured properly on the converged checkpoint,
scale *does* matter, though moderately — the same checkpoint ablated at its training
scale of 300 gives **+0.4612**, and at 160 gives **+0.4048**, a 12% loss. (An earlier
version of this README claimed layer-0 RMSNorm makes scale irrelevant. That is wrong:
RMSNorm normalizes the *attention* input, but the residual stream adds the raw
embedding back un-normalized, so magnitude propagates.) `--injection-scale` is
therefore required rather than defaulted in both `train_actor_sft.py` and
`train_rl.py`.

**Watch the ablation gap, not the eval CE.** Held-out CE plateaued at ~1.331 from
step 1000 onward while the ablation gap kept improving; trusting the CE curve would
have stopped training three checkpoints early. This matches the reference
implementation's own note: *"real-vs-rand gap is the signal, not train loss."*

### Injection scale

Measured on the same checkpoint (`actor_sft_8b_s50fix`, 256 held-out rows), varying
only the L2 norm the activation vector is rescaled to before replacing the marker
token's embedding:

| injection scale | real-vs-shuffled gap |
|---|---|
| **300** (matches SFT) | **+0.4612** |
| 160 | +0.4048 |

A ~2x scale error costs about **12%** of the conditioning signal — a real effect,
but not a dominant one. An earlier claim that layer-0 RMSNorm makes scale
irrelevant was wrong: RMSNorm normalises the *attention* input, but the residual
stream adds the raw embedding back un-normalised, so magnitude does propagate.

This matters because RL sidecars carry `injection_scale: null`, and the old
`2.5 * sqrt(d_model)` fallback silently produced 160 while the actor was SFT'd at
300. Both `train_actor_sft.py` and `train_rl.py` now **require** `--injection-scale`
rather than defaulting.

### Generated explanations

`scripts/actor_vector_ablation.py` proves the vector carries information; it says
nothing about whether the text is any good. `scripts/actor_gen_inspect.py` checks
the output directly on 500 held-out rows:

| | 4B actor (greedy) | 8B actor (greedy) | 8B actor (sampled) |
|---|---|---|---|
| unique explanations / 500 | 25 (5%) | **500 (100%)** | **500 (100%)** |
| content-word F1 vs own gold | — | 0.3819 | 0.3266 |
| content-word F1 vs other golds | — | 0.2267 | 0.1932 |
| retrieval acc (vs 19 distractors) | — | **68.4%** | **69.2%** |
| `<explanation>` extraction failures | 0% | 1.0% | 2.6% |

Mode collapse is gone. Overlap with the gold explanation is only meaningful
against a control — any two explanations of any vectors share words like
"vector"/"text"/"activations" — so each generation is also scored against *other*
rows' golds. Retrieval accuracy (own gold ranked above 19 random distractors) is
the more trustworthy metric, since correct paraphrase is lexically divergent and
token-F1 undercounts it.

A held-out sample. The model sees **only a 4096-dim vector** — never the source text:

```
GOLD: Immediate semantic expectations: The list format with hyphens and colons
      establishes a pattern where each line provides a specific statistic, so the
      next line must continue this structured data presentation.
      Syntactic/structural constraints: The last line "Nearest airport to White
      Plains: West" is an incomplete noun phrase; "West" is an adjective starting
      an airport name, requiring a completion like "Westchester County Airport".
      Final feature: The final token "West" is the initial word of an airport name.

GEN : Syntactic/structural constraint: The hyphenated list item "Airline: West" is
      incomplete, requiring a completion of the airline name (e.g., "Westchester
      County Airport").
      Immediate semantic expectation: The list format promises specific details for
      each bullet point, so the next token must continue the "Airline" field with
      the actual airline name or code.
      Final feature: The last token "West" is the start of a proper noun (airline
      name) within a bulleted list item, immediately requiring the remainder of that
      name to complete the entry.
```

It recovers the truncation point (`"West"`), the document's list structure, and the
completion (`Westchester County Airport`) from the vector alone. It misreads the
field label as "Airline" rather than "Nearest airport" — the kind of error that is
semantically close but lexically costly, and part of why F1 sits at ~0.38 while the
retrieval score is 68%.

### Known issues

- **Tag-extraction regression.** The 8B actor drops `<explanation>` tags on 1.0%
  (greedy) / 2.6% (sampled) of rows; the 4B actor never did. The GRPO reward path
  parses those tags, so this matters for RL.
### End-to-end coupling

Actor generates an explanation from a vector; critic reconstructs the vector from
that text. 500 held-out rows, `critic_sft_8b_s50` + `actor_sft_8b_s50fix`:

| decoding | FVE_nrm_meannorm | FVE_nrm | MSE | extraction failures |
|---|---|---|---|---|
| pre-fix actor (greedy) | −0.2793 | — | 0.8675 | 0% |
| **greedy** | **+0.5207** | +0.4228 | 0.3250 | 0.2% |
| **sampled T=1.0** | **+0.4581** | +0.3475 | 0.3675 | 1.0% |
| *gold explanations (ceiling)* | *+0.6165* | *+0.5382* | *0.2601* | — |

The loop closes: actor-generated text retains 84% (greedy) of what gold
API-written explanations achieve. GRPO rollouts are *sampled*, so +0.4581 — 74% of
the ceiling — is the baseline RL actually starts from.

## License

Apache-2.0

---

## Testing the models (for collaborators)

Weights live on the Hub, code lives here. You need both.

```bash
git clone git@github.com:TuxTu/tiny_nla.git && cd tiny_nla
pip install torch transformers accelerate huggingface_hub pyarrow numpy pyyaml
hf auth login          # the repo is private — you need access
```

Everything below downloads weights on first run (~16 GB per model, cached
afterwards) and needs one GPU with ≥24 GB. The two models are loaded and
freed in sequence, never both at once, so 24 GB is enough despite 2×16 GB of
downloads.

### The one command you need

`nla_demo.py` is the entry point. You choose the mode; it chooses the
checkpoint, always the current one on the Hub, with no flag to pin an older
one. It needs nothing from this repo, so you can also run it straight from the
Hub:

```bash
hf download TuHan/tiny-nla nla_demo.py --local-dir .

python nla_demo.py --mode code 'def gcd(a, b):
    while b: a, b = b, a % b
    return a'

python nla_demo.py --mode text 'The Federal Reserve announced yesterday that it
would raise interest rates, citing persistent inflation'
```

`--mode code` reconstructs a Python function; `--mode text` explains the
activation. The mode is explicit rather than sniffed from the input, because
guessing is wrong in both directions — a bare string literal is valid Python
but is not code, and a snippet with a typo is code but does not parse.
`--mode code` rejects input that is not valid Python instead of silently
falling back.

Add `--control` in either mode to also generate from a random vector. That is
the comparison that makes an output meaningful: it shows what the decoder's
prior produces with no information at all. Use `--file mymodule.py` for input
from a file.

**On a cluster (NSC/Berzelius).** The demo probes for a working C compiler on
startup and exports `CC` itself, because triton JIT-compiles a CUDA helper on
first GPU use and the `gcc` first on PATH here is a wrapper that refuses
without a build-env module. If you see a `CalledProcessError` naming a
`cuda_utils.c` you never wrote, set `CC=/usr/bin/gcc` by hand. Nothing needs to
go in `/tmp`.

The sections below are the same two paths with all the plumbing exposed — use
them if you want to point at a specific local checkpoint.

### Reconstruct a Python function from its activation vector

```bash
python scripts/nla_code_infer.py \
    --actor-ckpt TuHan/tiny-nla --subfolder code-decoder \
    --code 'def gcd(a, b):
    while b:
        a, b = (b, a % b)
    return a'
```

Prints the original, the reconstruction, and three scores. Add `--control` to
also generate from a random vector, which shows what the code prior produces
with no information — the comparison that makes the output meaningful.

Give it a file instead with `--file mymodule.py`.

**What to expect.** This is not a lossless codec. On a 300-function held-out set:

| | |
|---|---|
| exact string match | 1.0% |
| identical modulo renaming | 3.7% |
| AST-skeleton > 0.9 (near-identical shape) | 7.7% |
| retrieval@1 — is the output nearest its own target of 300? | **71%** |
| identifier recall | 17% |

So it reliably tells you *which* function a vector came from, usually gets the
shape and domain right, and usually gets the specific names wrong. The typical
error is a **sibling**: ask it for `send_video` and you get `send_document`,
same signature, same parameters, wrong entity.

### Explain a text activation vector

```bash
python scripts/nla_infer.py \
    --actor-ckpt TuHan/tiny-nla --actor-subfolder nla/actor \
    --critic-ckpt TuHan/tiny-nla --critic-subfolder nla/critic \
    --text "The Federal Reserve announced yesterday that it would raise rates"
```

Actor writes an explanation of the vector; critic reads the explanation back to
a vector; FVE scores the round trip. Gold-explanation ceiling is +0.6165.

### What is in the Hub repo

```
TuHan/tiny-nla
  nla/actor/         text actor, 3 epochs on 124,741 rows, conditioning gap +0.4593
  nla/critic/        critic, 65.7% held-out FVE
  code-decoder/      code reconstruction  + centre_mean.npy  + nla_meta.yaml
```

**`code-decoder` requires `centre_mean.npy`.** It was trained on mean-centred
vectors, and the offset carries ~82% of a typical vector's magnitude — inject a
raw vector and the model produces fluent, plausible, completely unrelated code.
`nla_code_infer.py` fetches it automatically when you pass `--subfolder`; if you
load the weights yourself, subtract it before injecting.

### Reading the numbers

Every metric should be compared against its control, not against zero:

- `--control` / shuffled vector is the floor. For round-trip FVE that floor is
  about **−0.78**, not 0, so a raw score of 0.14 can still be most of the
  available signal.
- Two arbitrary Python functions already share **~0.52** AST-skeleton
  similarity and **~0.15** token overlap. Only the margin above that is real.

Full results, method and failure analysis: see the experiment report.
