"""Data generation pipeline — corpus → positions → labels → training parquets.

Stage 0: extract text positions (tokenizer-only, CPU)
Stage 1: document-level split into AV-SFT / AR-SFT / RL subsets
Stage 2: API labeling with DeepSeek or Anthropic

All stages are model-agnostic — they work with decoded text, never
activation vectors. Labels produced once are reusable across all models
sharing the same tokenizer family.
"""
