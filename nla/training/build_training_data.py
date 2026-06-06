"""Build training-ready parquets by joining explained + vectors on (doc_id, n_raw_tokens).

Produces AV-SFT, AR-SFT, and RL format parquets matching the original stage3_build schema,
plus sidecar YAML with full token metadata.
"""

import argparse
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from nla.training.injection_tokens import build_token_meta
from nla.training.schema import (
    ACTIVATION_COLUMN,
    INJECT_PLACEHOLDER,
    wrap_explanation,
)
from nla.training.sidecar import write_dataset_sidecar

# ---- prompt templates (matching original) -----------------------------------

_DEFAULT_ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. "
    "You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets "
    "describing that vector.\n\n"
    "Here is the vector:\n\n"
    "<concept>{injection_char}</concept>\n\n"
    "Please provide an explanation."
)

_DEFAULT_CRITIC_TEMPLATE = (
    "Summary of the following text: <text>{explanation}</text> <summary>"
)

_PROMPT_STRUCT = pa.struct([("role", pa.string()), ("content", pa.string())])


def _validate_ar_sft_tail(prompts: list[str], suffix_ids: list[int], tokenizer) -> None:
    """Assert every AR-SFT prompt ends with the expected suffix token IDs."""
    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        tail = ids[-len(suffix_ids):]
        assert tail == suffix_ids, (
            f"Row {i}: AR-SFT prompt tail {tail} != expected suffix {suffix_ids}. "
            f"This means the BPE boundary merged unexpectedly with the explanation. "
            f"Prompt tail: {tokenizer.decode(tail)!r}"
        )


def build(
    explained_path: str,
    vectors_path: str,
    tokenizer,
    split_type: str,
    output_path: str,
    actor_template: str = _DEFAULT_ACTOR_TEMPLATE,
    critic_template: str = _DEFAULT_CRITIC_TEMPLATE,
    d_model: int | None = None,
) -> None:
    """Join explained + vectors, format prompts, write training parquet + sidecar."""
    # ---- load and join -------------------------------------------------------
    expl = pq.read_table(explained_path)
    vecs = pq.read_table(vectors_path)

    # Verify join keys match
    expl_keys = set(
        zip(expl.column("doc_id").to_pylist(), expl.column("n_raw_tokens").to_pylist())
    )
    vec_keys = set(
        zip(vecs.column("doc_id").to_pylist(), vecs.column("n_raw_tokens").to_pylist())
    )
    missing = expl_keys - vec_keys
    extra = vec_keys - expl_keys
    assert not missing, f"{len(missing)} rows in explained missing from vectors"
    assert not extra, f"{len(extra)} rows in vectors not in explained"

    if d_model is None:
        d_model = len(vecs.column(ACTIVATION_COLUMN)[0].as_py())
    print(f"d_model={d_model}  rows={expl.num_rows}  split_type={split_type}")

    # ---- token metadata ------------------------------------------------------
    needs_critic = split_type == "ar_sft"
    token_meta = build_token_meta(
        tokenizer,
        actor_template,
        critic_template=critic_template if needs_critic else None,
    )
    print(f"injection: char={token_meta.injection_char!r}  id={token_meta.injection_token_id}")
    print(f"  neighbors: left={token_meta.injection_left_neighbor_id}  right={token_meta.injection_right_neighbor_id}")
    if needs_critic:
        print(f"  critic suffix ids: {token_meta.critic_suffix_ids}")

    # ---- format prompts ------------------------------------------------------
    n = expl.num_rows

    if split_type in ("av_sft", "rl"):
        # Actor template with <INJECT> placeholder (swapped at training time)
        prompt_content = actor_template.format(injection_char=INJECT_PLACEHOLDER)
        prompt_msg = [{"role": "user", "content": prompt_content}]
        prompt_col = pa.array([prompt_msg] * n, type=pa.list_(_PROMPT_STRUCT))

    if split_type == "av_sft":
        api_explanations = expl.column("api_explanation").to_pylist()
        response_col = pa.array(
            [wrap_explanation(e) for e in api_explanations], type=pa.string()
        )
        table_dict = {
            "prompt": prompt_col,
            "response": response_col,
            ACTIVATION_COLUMN: vecs.column(ACTIVATION_COLUMN),
        }

    elif split_type == "ar_sft":
        api_explanations = expl.column("api_explanation").to_pylist()
        prompts = [critic_template.format(explanation=e) for e in api_explanations]
        _validate_ar_sft_tail(prompts, token_meta.critic_suffix_ids, tokenizer)
        table_dict = {
            "prompt": pa.array(prompts, type=pa.string()),
            ACTIVATION_COLUMN: vecs.column(ACTIVATION_COLUMN),
        }

    elif split_type == "rl":
        table_dict = {
            "prompt": prompt_col,
            ACTIVATION_COLUMN: vecs.column(ACTIVATION_COLUMN),
        }

    else:
        raise ValueError(f"unknown split_type={split_type!r}")

    # Pass-through provenance columns
    for col_name in ("doc_id", "n_raw_tokens", "activation_layer"):
        if col_name in expl.column_names:
            table_dict[col_name] = expl.column(col_name)

    table = pa.table(table_dict)
    pq.write_table(table, output_path)
    print(f"wrote {n} rows → {output_path}")

    # ---- sidecar -------------------------------------------------------------
    write_dataset_sidecar(
        output_path,
        d_model=d_model,
        token_meta=token_meta,
        split_type=split_type,
        actor_template=actor_template,
        critic_template=critic_template if needs_critic else None,
        injection_scale=None,   # raw vectors — training decides
        mse_scale="sqrt_d_model",
        num_rows=n,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explained", required=True, help="explained parquet path")
    p.add_argument("--vectors", required=True, help="vectors parquet path")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer name")
    p.add_argument("--output", required=True, help="output training parquet path")
    p.add_argument("--split-type", required=True,
                   choices=["av_sft", "ar_sft", "rl"])
    p.add_argument("--actor-template", default=_DEFAULT_ACTOR_TEMPLATE)
    p.add_argument("--critic-template", default=_DEFAULT_CRITIC_TEMPLATE)
    p.add_argument("--d-model", type=int, default=None)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    build(
        explained_path=args.explained,
        vectors_path=args.vectors,
        tokenizer=tokenizer,
        split_type=args.split_type,
        output_path=args.output,
        actor_template=args.actor_template,
        critic_template=args.critic_template,
        d_model=args.d_model,
    )


if __name__ == "__main__":
    main()
