"""NLA roundtrip interface — load once, explain and reconstruct interactively.

Usage:
    nla = NLA(
        base_model="Qwen/Qwen3-0.6B",
        actor_ckpt="checkpoints/actor_sft",
        critic_ckpt="checkpoints/critic",
    )
    explanation = nla.explain("The quick brown fox jumps over the lazy dog", position=4)
    reconstructed = nla.reconstruct(explanation)
    mse, cos = nla.score(explanation, text="The quick brown fox jumps over the lazy dog", position=4)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.training.injection import inject_at_marked_positions
from nla.training.models import NLACriticModel
from nla.training.schema import extract_explanation, normalize_activation
from nla.training.sidecar import read_sidecar

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


class NLA:
    """End-to-end NLA pipeline: extraction, verbalization, reconstruction, scoring.

    Parameters
    ----------
    base_model :
        HF model name or path for activation extraction. Must match the tokenizer
        family used during labeling (e.g. Qwen/Qwen3-4B).
    actor_ckpt :
        Actor SFT checkpoint dir (contains config.json, nla_meta.yaml).
    critic_ckpt :
        Critic checkpoint dir (contains value_head.safetensors, nla_meta.yaml).
    data :
        Path to a training parquet with a sidecar, OR a sidecar YAML path,
        OR a pre-loaded sidecar dict. Used only for injection token metadata.
    layer_index :
        Transformer layer for activation extraction. Default: 2/3 of total layers.
    device :
        "cuda", "mps", or "cpu".
    dtype :
        Model dtype (bfloat16 recommended for CUDA).
    """

    def __init__(
        self,
        *,
        base_model: str,
        actor_ckpt: str,
        critic_ckpt: str,
        data: str | dict | None = None,
        layer_index: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        # ---- sidecar (injection params) -------------------------------------
        if data is None:
            data = actor_ckpt
        if isinstance(data, dict):
            sidecar = data
        else:
            sidecar = read_sidecar(data)
        tokens = sidecar.get("tokens", {})
        self.inj_char = tokens["injection_char"]
        self.inj_id = tokens["injection_token_id"]
        self.left_id = tokens["injection_left_neighbor_id"]
        self.right_id = tokens["injection_right_neighbor_id"]
        self.d_model = sidecar.get("extraction", {}).get("d_model")
        self.mse_scale = math.sqrt(self.d_model) if self.d_model else 32.0
        self.critic_template = sidecar.get("prompt_templates", {}).get(
            "critic",
            "Summary of the following text: <text>{explanation}</text> <summary>",
        )

        # ---- tokenizer ------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(actor_ckpt)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # ---- actor ----------------------------------------------------------
        print(f"loading actor from {actor_ckpt} ...")
        self.actor = AutoModelForCausalLM.from_pretrained(
            actor_ckpt, torch_dtype=dtype,
            device_map={"": self.device} if device == "cuda" else None,
        )
        if device == "mps":
            self.actor = self.actor.to(self.device)
        self.actor.eval()
        self.actor_embed = self.actor.get_input_embeddings()
        if self.d_model is None:
            self.d_model = self.actor.config.hidden_size
            self.mse_scale = math.sqrt(self.d_model)
        print(f"  d_model={self.d_model}  layers={self.actor.config.num_hidden_layers}")

        # ---- critic ---------------------------------------------------------
        print(f"loading critic from {critic_ckpt} ...")
        self.critic = NLACriticModel.from_pretrained(
            critic_ckpt, torch_dtype=dtype,
            device_map={"": self.device} if device == "cuda" else None,
        )
        if device == "mps":
            self.critic = self.critic.to(self.device)
        self.critic.eval()
        print(f"  critic: {self.critic.config.num_hidden_layers} layers")

        # ---- base model (extraction) ----------------------------------------
        print(f"loading base model {base_model} ...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype,
            device_map={"": self.device} if device == "cuda" else None,
        )
        if device == "mps":
            self.base_model = self.base_model.to(self.device)
        self.base_model.eval()
        self.base_layers = self.base_model.model.layers
        if layer_index is None:
            layer_index = (2 * len(self.base_layers)) // 3
        self.layer_index = layer_index
        print(f"  extraction layer: {self.layer_index}/{len(self.base_layers)}")

        # ---- prompt template ------------------------------------------------
        self.prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self._build_prompt_content()}],
            tokenize=False, add_generation_prompt=True,
        )

    def _build_prompt_content(self) -> str:
        return (
            "You are a meticulous AI researcher conducting an important investigation "
            "into activation vectors from a language model. Your overall task is to "
            "describe the semantic content of that activation vector.\n\n"
            "We will pass the vector enclosed in <concept> tags into your context. "
            "You must then produce an explanation for the vector, enclosed within "
            "<explanation> tags. The explanation consists of 2-3 text snippets "
            "describing that vector.\n\n"
            "Here is the vector:\n\n"
            f"<concept>{self.inj_char}</concept>\n\n"
            "Please provide an explanation."
        )

    # ---- extraction ---------------------------------------------------------

    @torch.no_grad()
    def extract(self, text: str, position: int) -> torch.Tensor:
        """Extract activation vector at a token position.

        Parameters
        ----------
        text : str
            Input text to encode.
        position : int
            0-indexed token position (post-tokenization).

        Returns
        -------
        activation : [d_model] float32 tensor on CPU.
        """
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048,
        ).to(self.device)
        pos = min(position, enc["input_ids"].shape[1] - 1)

        captured: torch.Tensor | None = None

        def _hook(_module, _inputs, output):
            nonlocal captured
            captured = output[0].detach().clone()

        handle = self.base_layers[self.layer_index].register_forward_hook(_hook)
        self.base_model(input_ids=enc["input_ids"], use_cache=False)
        handle.remove()

        assert captured is not None, "extraction hook did not fire"
        return captured[0, pos].float().cpu()

    # ---- verbalization ------------------------------------------------------

    @torch.no_grad()
    def explain(
        self,
        vector: torch.Tensor | np.ndarray | None = None,
        *,
        text: str | None = None,
        position: int = 0,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """Generate an explanation from an activation vector.

        Pass either ``vector`` directly, or ``text`` + ``position`` to extract
        and explain in one call.

        Returns the full generated text (including <explanation> tags).
        """
        if vector is None:
            assert text is not None, "provide either vector= or text= + position="
            vector = self.extract(text, position)

        v = torch.as_tensor(np.asarray(vector, dtype=np.float32)).to(
            self.device, dtype=self.dtype,
        ).unsqueeze(0)  # [1, d]

        enc = self.tokenizer(self.prompt_text, return_tensors="pt").to(self.device)

        def _make_hook(_vec):
            def _hook_fn(_module, _args, output):
                if output.shape[1] > 1:
                    return inject_at_marked_positions(
                        _args[0], output, _vec, self.inj_id,
                        self.left_id, self.right_id,
                    )
                return output
            return _hook_fn

        hook = self.actor_embed.register_forward_hook(_make_hook(v))
        try:
            gen_out = self.actor.generate(
                enc["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            hook.remove()

        return self.tokenizer.decode(
            gen_out[0, enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    # ---- reconstruction -----------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        """Reconstruct activation vector from explanation text.

        Parameters
        ----------
        explanation : str
            Raw explanation text (with or without <explanation> tags).

        Returns
        -------
        vector : [d_model] float32 tensor on CPU.
        """
        # Strip <explanation> tags if present
        m = EXPLANATION_RE.search(explanation)
        text = m.group(1).strip() if m else explanation.strip()

        prompt = self.critic_template.format(explanation=text)
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)

        c_out = self.critic(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"])
        seq_len = enc["attention_mask"].sum(dim=1) - 1
        return c_out.values[0, seq_len[0]].float().cpu()

    # ---- scoring ------------------------------------------------------------

    def score(
        self,
        explanation: str,
        vector: torch.Tensor | np.ndarray | None = None,
        *,
        text: str | None = None,
        position: int = 0,
    ) -> tuple[float, float]:
        """Compute roundtrip (MSE, cosine_similarity).

        Returns direction-only MSE (2(1-cos), range [0,4]) and cosine similarity.
        Perfect reconstruction → MSE=0, cos=1.0.
        Orthogonal → MSE=2.0, cos=0.0.
        """
        if vector is None:
            assert text is not None, "provide either vector= or text= + position="
            vector = self.extract(text, position)

        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(vector, dtype=np.float32))

        pn = normalize_activation(pred.unsqueeze(0), self.mse_scale)
        gn = normalize_activation(gold.unsqueeze(0), self.mse_scale)
        mse = float(F.mse_loss(pn, gn).item())
        cos = float(F.cosine_similarity(
            pred.unsqueeze(0).float(), gold.unsqueeze(0).float(),
        ).item())
        return mse, cos

    # ---- batch evaluation ---------------------------------------------------

    def evaluate_batch(
        self,
        texts: list[str],
        positions: list[int],
        *,
        max_new_tokens: int = 200,
    ) -> list[dict]:
        """Run full roundtrip on a batch: extract → explain → reconstruct → score.

        Returns list of dicts with keys: text, position, explanation, mse, cos.
        """
        results = []
        for i, (text, pos) in enumerate(zip(texts, positions)):
            vec = self.extract(text, pos)
            explanation = self.explain(vec, max_new_tokens=max_new_tokens)
            clean = extract_explanation(explanation)
            record = {
                "idx": i, "text": text[:200], "position": pos,
                "explanation": clean, "raw_output": explanation,
            }
            if clean is not None:
                mse, cos = self.score(clean, vec)
                record["mse"] = mse
                record["cos"] = cos
            else:
                record["mse"] = None
                record["cos"] = None
            results.append(record)
        return results
