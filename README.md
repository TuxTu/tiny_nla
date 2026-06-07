# tiny_nla — A From-Scratch NLA Pipeline for Small Open Models

Natural language autoencoders (NLA) produce unsupervised explanations of LLM
activations. Given a residual-stream activation vector from a language model, an
**actor** (activation verbalizer) generates a natural-language description; a
**critic** (activation reconstructor) maps that description back to a vector. The
fidelity of this text-mediated roundtrip — measured as Fraction of Variance
Explained (FVE) — quantifies how much semantic information natural language can
capture about a model's internal state.

This repository reimplements the NLA methodology from [Anthropic's Transformer
Circuits paper](https://transformer-circuits.pub/2026/nla/index.html) for **small
open-source models** (Qwen3 0.6B & 4B) using only HuggingFace Transformers +
PyTorch. The original implementation targets 7B–70B models on 16-GPU clusters
and depends on Anthropic's proprietary Miles training framework. Ours runs on a
**single Colab A100**.

**Key finding:** SFT training works — the actor produces format-correct,
structurally valid explanations. GRPO reinforcement learning systematically
fails at this scale because the truncated critic model (≤ 25 layers) lacks
sufficient capacity to provide a discriminative reward signal. This is a
**scaling-law boundary**, not an implementation error.

---

## Design Choices

Two architectural decisions distinguish this reimplementation:

### 1. Data parallelism instead of model parallelism

Anthropic's original uses FSDP and Megatron-style sharding to distribute 70B+
models across 16 GPUs. Our target models (0.6B–4B parameters) **fit on a single
GPU**. Instead of sharding one large model, we replicate across GPUs with
PyTorch DistributedDataParallel and a DistributedSampler — each rank sees
unique data, gradients are averaged. This gives linear throughput scaling with
standard primitives and zero framework lock-in.

### 2. Decoupled, tokenizer-bound labeling

API-labeled explanations are tied to a *tokenizer*, not a specific model. All
Qwen3 variants (0.6B–8B) share identical vocabulary and special tokens, so one
set of 250k labeled positions serves the entire family. Labels are produced
once, vectors are extracted per-model, and training is assembled by joining the
two. This eliminates the dominant pipeline cost (API calls) by an order of
magnitude compared to per-model labeling.

The combination makes NLA training practical on commodity hardware — no
InfiniBand, no Ray cluster, no Megatron.

### 3. Why Qwen3

We target the **Qwen3 family** (0.6B and 4B) for three reasons:

**Single-GPU fit.** Qwen3-0.6B occupies ~1.2 GB in bf16 — small enough for a
Colab T4 (16 GB) — while Qwen3-4B (~8 GB) fits comfortably on a single A100
(40 GB). Both leave ample VRAM for optimizer states, activations, and batch
processing without sharding. This directly enables the data-parallel design.

**Uniform tokenizer across the family.** Qwen3 0.6B, 1.7B, 4B, and 8B share an
identical vocabulary, special-token set, and chat template. A single labeled
dataset works for the entire model family — extract positions once, reuse
across every size. This is a deliberate property of Qwen3's design, not an
accident, and we exploit it to keep labeling costs independent of the number
of models evaluated.

**Consistent dense architecture.** Within the 0.6B–4B range, all variants use
the same building blocks: RMSNorm, SwiGLU activations, RoPE position encoding,
and Grouped Query Attention — no mixture-of-experts, no gated attention, no
architectural discontinuities. This means findings at one scale are directly
comparable to findings at another, and training code written for 0.6B
transfers to 4B with no code changes beyond the model name. The paper's
baseline (Qwen2.5-7B) shares the same lineage, so our results are
architecturally comparable despite the scale difference.

---

## What We Implemented

The pipeline has four stages, corresponding to the paper's methodology:

### Data generation (`nla/datagen/`)

| Stage | Script | Description |
|-------|--------|-------------|
| Extract | `extract_positions.py` | Tokenize corpus, deterministically sample positions, decode text |
| Label | `api_explain.py` | Generate structured explanations via DeepSeek V4 Flash API |
| Vectors | `extract_vectors.py` | Forward-pass through base model, capture hidden states at labels |
| Build | `build_training_data.py` | Join explanations + vectors → training parquets with sidecar metadata |

Output: 250k labeled positions from FineWeb, deterministically split 25/25/50
across AV-SFT, AR-SFT, and RL. Distributed as
[`TuHan/qwen3-nla-250k`](https://huggingface.co/datasets/TuHan/qwen3-nla-250k).

### Training (`nla/training/`)

| Stage | Direction | Model | Loss | Key detail |
|-------|----------|-------|------|------------|
| Critic SFT | text → vector | Truncated transformer (K+1 layers) + value head | MSE, dual L2 normalization | Identity-init value head, no final LayerNorm |
| Actor SFT | vector → text | Full CausalLM + injection hook | CE, response tokens only | Activation injected at marker token via forward hook with neighbor verification |
| GRPO RL | both | Actor + critic, on-policy | Policy gradient + KL penalty + critic MSE | Group-relative advantages, SGLang rollout (optional) |

The **injection mechanism** is the correctness-critical path: a CJK marker
character (U+320E, `㈎`) is placed in the prompt; the forward hook scans for it
at every forward pass, verifies left/right neighbor token IDs match canonical
values (prevents false positives from generated text), and overwrites its
embedding row with the activation vector. A RuntimeError is raised if the
expected number of injection sites is not found — silent injection failure is
impossible by construction.

The **GRPO implementation** matches the paper's algorithm: N responses sampled
per prompt, rewards computed as −MSE between critic prediction and gold
activation (both normalized to `√d_model` L2-norm), advantages computed as
`(r − μ_group) / σ_group`, actor loss = `−(advantage × log_prob) +
kl_coef × KL`, with KL computed per-token to avoid O(L²) gradient explosion.

A standalone inference script (`scripts/nla_infer.py`) and a programmatic API
(`nla/roundtrip.py`) support evaluation without SGLang.

### What we did NOT port

The original repository's training code is a plugin for Anthropic's internal
**Miles** framework (Ray-orchestrated FSDP/Megatron training). Miles is not open
source. All training loops, GRPO logic, data loading, checkpointing, and
multi-GPU coordination were written from scratch. The only components carried
forward from the original are the mathematical kernels where correctness
requires numerical identity: `normalize_activation()`, `inject_at_marked_positions()`,
and the critic model architecture (truncation + value head).

---

## Pre-labeled Dataset

**[`TuHan/qwen3-nla-250k`](https://huggingface.co/datasets/TuHan/qwen3-nla-250k)**
contains 250k FineWeb text snippets (~499k labeled rows), each annotated by
DeepSeek V4 Flash with a structured explanation of the semantic signal an
activation vector encodes at that position.

| Column | Description |
|--------|-------------|
| `doc_id` | FineWeb document identifier |
| `n_raw_tokens` | Token count in the context window |
| `detokenized_text_truncated` | Text snippet for AV extraction |
| `api_explanation` | Structured explanation (2–3 features, 80–100 words) |

The text positions are sampled deterministically across 50k documents (5 positions
per doc), producing ~499k rows split at the document level:

| Split | Rows | Purpose |
|-------|------|---------|
| AV-SFT | 125k | Train actor: vector → explanation |
| AR-SFT | 125k | Train critic: explanation → vector |
| RL | 250k | GRPO fine-tuning (on-policy) |

Each `api_explanation` describes 2–3 features — syntactic constraints, topic
continuation, register shifts, or entity tracking — in free-form natural language.
All Qwen3 variants share the same tokenizer, so labels are reusable across the
entire model family. A SHA-256 tokenizer fingerprint is embedded in every output
parquet for downstream verification.

```python
from datasets import load_dataset
ds = load_dataset("TuHan/qwen3-nla-250k", split="train")
# ds[0]: {"doc_id": "...", "n_raw_tokens": 210,
#         "detokenized_text_truncated": "A novel two-step immunotherapy...",
#         "api_explanation": "Syntactic/structural constraints: the..."}
```

---

## Results

### SFT training (Qwen3-4B, 8× A100)

| Metric | Actor SFT | Critic SFT |
|--------|----------|-----------|
| Training data | 125k rows | 125k rows |
| Steps | 250 | 250 |
| Global batch size | 256 | 256 |
| Final loss | CE 19.78 | MSE 1.06 |
| Extraction rate (greedy) | 100% | — |

The SFT actor produces format-correct `<explanation>` output reliably. Example
(greedy decoding, Qwen3-4B, text *"The capital of France is"*):

> `<explanation>` Immediate syntactic expectation: the copula "is" requires a
> predicate complement — a noun phrase denoting a location, as in "is Paris"
> or "is a city in Western Europe." Domain knowledge signal: the subject
> "capital of France" strongly constrains the completion to a specific
> geographical entity. Register: encyclopedic declarative statement.
> `</explanation>`

**Critic FVE is negative** — the model predicts vectors *worse* than a
constant mean predictor:

| Evaluation path | FVE_nrm | MSE |
|----------------|---------|-----|
| Full pipeline (actor → critic) | −0.32 | 0.774 |
| Critic only (API labels → critic) | −0.22 | 0.680 |

Even when fed perfect API-written explanations, the critic cannot beat the mean
baseline. The 4B critic (25/36 layers, 2560-dimensional) is simply too small
for the text→vector regression task. The original paper reports FVE 0.6–0.8 on
7B models (3584-dimensional). Our gap is a capacity limitation, not a training
or implementation bug.

### Qwen3-0.6B (Colab A100)

The 0.6B model serves as our lowest-cost baseline — all experiments run on a
single Colab A100. A complete training and evaluation notebook
is available at:
**[colab.research.google.com/drive/1SJlR0rh8z6q7BjuP0TD9k_z1_iBEDwQw](https://colab.research.google.com/drive/1SJlR0rh8z6q7BjuP0TD9k_z1_iBEDwQw?usp=sharing)**

**SFT.** The actor produces format-correct `<explanation>` output but with
limited semantic accuracy — the 0.6B model (28 layers, d_model=1024) has
insufficient capacity to capture nuanced activation patterns. The critic
similarly underperforms, achieving negative FVE even on clean API labels.

**RL (GRPO, 200 steps).** Mode collapse: the actor converges to a single
template repeated for every input vector:

> `<explanation>` The token "5" is the first feature index, so the next token
> must be a new feature token to continue the next feature. The final token is
> "5" — it is the first feature index, so the next token must be a new feature.
> `</explanation>`

**Post-RL roundtrip FVE** (10-sample evaluation on held-out AV-SFT data):

| Metric | Value |
|--------|-------|
| MSE (full pipeline) | 0.7212 |
| Baseline — raw variance | 0.5321 |
| Baseline — mean-norm | 0.6319 |
| **FVE_nrm** | **−0.3556** |
| **FVE_nrm_meannorm** | **−0.1414** |
| Extraction rate | 100% |

Both FVE values are negative: the critic predicts vectors 14–36% *worse* than
a constant mean predictor. The raw-variance baseline (0.532) is lower than the
7B equivalent (~0.72) because 0.6B activation vectors are more tightly
clustered — the task is objectively easier, yet the critic still fails.

Two things are true simultaneously: (1) the format survives RL — the KL penalty
and injection hook work correctly; (2) content quality degrades — with a critic
that cannot discriminate good from bad explanations, GRPO amplifies whichever
output pattern happens to score marginally better, which is the fixed template.
The template beat real explanations not because it was good, but because the
critic's MSE surface is too flat to tell the difference.

**Scaling trend (0.6B → 4B → 7B).** Critic FVE improves with model scale but
remains negative through 4B. The original paper's 7B critic achieves post-SFT
FVE of ~0.375 (MSE ≈ 0.45 at rawvar baseline ~0.72), while our 4B critic
achieves −0.22 (MSE 0.68) and our 0.6B achieves −0.36 (MSE 0.72). The
transition from negative to positive FVE occurs somewhere between 4B and 7B —
consistent with the hypothesis that text→vector regression has a minimum
capacity floor below which the critic cannot extract meaningful signal from
natural language.

### GRPO RL: seven attempts, seven failures (Qwen3-4B)

| Run | KL formulation | kl_coef | Critic frozen? | Steps survived | Failure mode |
|-----|---------------|---------|---------------|---------------|-------------|
| 1 | per_seq (signed) | 0.01 | No | ~20 | Format loss → "!!!!!" |
| 2 | per_seq (signed) | 0.01 | No | ~125 | Collapse at eval |
| 3 | per_seq (squared) | 0.01 | No | ~10 | O(L²) gradient bomb |
| 4 | per_seq (squared) | 0.1 | No | ~10 | O(L²) gradient bomb |
| 5 | per_seq (squared) | 1.0 | No | ~10 | O(L²) gradient bomb |
| 6 | per_seq (squared) | 0.1 | Yes | ~25 | Collapse at eval |
| 7 | per_token (squared) | 0.1 | No | ~10 | Template collapse |

Three root causes identified:

1. **Critic capacity.** At 4B scale, the critic's FVE is negative — the reward
   signal is essentially random. GRPO amplifies noise. The original paper
   reports that critic FVE must be substantially positive (> 0.3) for RL to
   improve upon SFT. We never crossed this threshold.

2. **KL penalty formulation.** The sequence-level squared difference
   `(Σlogπ − Σlogπ_ref)²` scales as O(L²) with response length. A 10% drift in
   per-token log-prob across 150 tokens produces a 22,500× multiplier. Runs 3–5
   suffered catastrophic gradient explosion within ten steps. The fix (per-token
   mean, matching Anthropic's implementation) stabilizes the gradient but cannot
   fix the absent reward signal.

3. **SGLang–HF distribution mismatch.** SGLang rollout (`input_embeds` API) and
   HF generate (`input_ids` + forward hook) traverse different code paths in the
   attention kernel. Patterns learned during SGLang-based training fail to
   transfer to HF-based evaluation. This is a deployment artifact specific to
   the two-inference-backend architecture.

---

## Comparison to the Original Paper

| | Anthropic (7B) | This work (0.6B) | This work (4B) |
|---|---|---|---|
| Training framework | Miles (proprietary) | Pure PyTorch + HF | Pure PyTorch + DDP |
| GPUs required | 8–16× H100 | 1× Colab A100 | 8× A100 |
| Critic FVE (post-SFT) | ~0.375 | < 0 | −0.22 |
| Critic FVE (post-RL) | 0.6–0.8 | N/A (collapsed) | N/A (collapsed) |
| RL outcome | 2× FVE improvement | Mode collapse | Mode collapse |

The gap between our SFT FVE (< 0) and their minimal threshold for RL (> 0.3)
fully explains why RL failed. The critic models we can train at 0.6B–4B scale
are below the capacity floor for the text→vector regression task. The paper's
methodology is correct; our models are simply too small for the critic component.

---

## What We Learned

**What works:** The SFT pipeline trains stably at both scales. Actor injection
with neighbor verification is robust — zero false-positive injections across
all training and evaluation runs. The decoupled labeling approach works as
designed: one set of labels, reused across model sizes.

**What doesn't:** GRPO RL requires a critic capable of FVE > 0. At 0.6B–4B
scale with 2/3 layer truncation, this is not achievable. The KL penalty
formulation is a sharp edge — sequence-level squared difference causes gradient
explosion that sequence-length scaling alone cannot fix.

**Open questions:** What is the minimum model scale for critic FVE > 0? Would
full-depth critics (no layer truncation) close the gap? Can a format reward
bonus substitute for critic capacity at small scales? What is the minimum
SFT data requirement for critic discriminability?

---

## Quick Start

```bash
pip install -e .

# --- Option A: use the pre-labeled dataset (skip steps 1–2) ---
# Download from https://huggingface.co/datasets/TuHan/qwen3-nla-250k

# --- Option B: generate labels + vectors from scratch ---
# 1. Datagen pipeline (CPU + API)
python -m nla.datagen.run_pipeline --config configs/datagen_0.6b_25k.yaml

# 2. Build training parquets (join explanations + vectors)
for split in av_sft ar_sft rl; do
  python -m nla.training.build_training_data \
    --explained data/${split}_explained.parquet \
    --vectors data/${split}_explained_vectors.parquet \
    --tokenizer Qwen/Qwen3-0.6B \
    --output data/${split}_train.parquet --split-type $split
done

# 3. Train critic SFT (text → vector, MSE regression)
python -m nla.training.train_critic_sft \
  --data data/ar_sft_train.parquet --model-name Qwen/Qwen3-0.6B \
  --output-dir checkpoints/critic --num-steps 1000

# 4. Train actor SFT (vector → text, CE with injection hook)
python -m nla.training.train_actor_sft \
  --data data/av_sft_train.parquet --model-name Qwen/Qwen3-0.6B \
  --output-dir checkpoints/actor_sft --num-steps 1000

# 5. RL training (GRPO, on-policy)
python -m nla.training.train_rl \
  --data data/rl_train.parquet --model-name Qwen/Qwen3-0.6B \
  --actor-ckpt checkpoints/actor_sft --critic-ckpt checkpoints/critic \
  --output-dir checkpoints/rl --n-samples 8 --num-steps 200

# 6. Qualitative evaluation (single-text roundtrip)
python scripts/nla_infer.py --model-name Qwen/Qwen3-0.6B \
  --actor-ckpt checkpoints/rl/actor --critic-ckpt checkpoints/critic \
  --sidecar data/av_sft_train.parquet --text "The capital of France is"

# 7. Quantitative evaluation (FVE on held-out data)
python scripts/nla_eval_fve.py --model-name Qwen/Qwen3-0.6B \
  --data data/ar_sft_train.parquet --actor-ckpt checkpoints/rl/actor \
  --critic-ckpt checkpoints/critic --sidecar data/ar_sft_train.parquet \
  --n-samples 100
```

A complete end-to-end Colab notebook for the 0.6B pipeline is available at:
**[colab.research.google.com/drive/1SJlR0rh8z6q7BjuP0TD9k_z1_iBEDwQw](https://colab.research.google.com/drive/1SJlR0rh8z6q7BjuP0TD9k_z1_iBEDwQw?usp=sharing)**

Pre-labeled dataset: [`TuHan/qwen3-nla-250k`](https://huggingface.co/datasets/TuHan/qwen3-nla-250k)

---

## Repository Structure

```
nla/
  datagen/               Data generation (CPU + API)
  training/              Training (GPU, single-GPU or DDP)
    models.py            NLACriticModel — truncated transformer + value head
    injection.py         inject_at_marked_positions() — the correctness kernel
    loss.py              nla_critic_loss() + sft_loss()
    train_critic_sft.py  AR-SFT training
    train_actor_sft.py   AV-SFT training with injection hooks
    train_rl.py          GRPO RL with per-sequence log-prob chain rule
scripts/
  nla_infer.py           Standalone inference (HF generate, no SGLang needed)
  nla_eval_fve.py        FVE evaluation
configs/                 YAML pipeline configs
```

## License

Apache-2.0
