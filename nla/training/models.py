"""NLACriticModel: truncated transformer + vector value head.

Ported from original nla/models.py. Simplified for single-GPU, Qwen3-only.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


@dataclass
class NLACriticOutput:
    values: torch.Tensor  # [B, T, d_model]
    backbone_last_hidden: torch.Tensor  # [B, T, d_model]


def _truncate_config_layers(config, num_layers: int) -> None:
    """Set num_hidden_layers and truncate per-layer arrays to match."""
    config.num_hidden_layers = num_layers
    for attr in ("layer_types", "sliding_window_pattern", "no_rope_layers"):
        v = getattr(config, attr, None)
        if isinstance(v, (list, tuple)) and len(v) > num_layers:
            setattr(config, attr, type(v)(v[:num_layers]))


def _inner_transformer(backbone: PreTrainedModel) -> nn.Module:
    """Get the inner transformer module (.model for Qwen/Llama)."""
    if hasattr(backbone, "model"):
        return backbone.model
    raise AssertionError(f"{type(backbone).__name__} has no .model attribute")


class NLACriticModel(PreTrainedModel):
    """Truncated transformer + linear value head (d_model → d_model, no bias).

    K+1 layers where K = extraction layer index. No final layernorm — raw
    residual stream goes to value head. Identity-initialized.
    """

    def __init__(self, config, backbone: PreTrainedModel):
        super().__init__(config)
        self.backbone = backbone
        self.value_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self._no_split_modules = getattr(backbone, "_no_split_modules", [])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, nla_num_layers: int | None = None, **kwargs):
        """Load with truncation to nla_num_layers + 1 transformer blocks."""
        kwargs.setdefault("trust_remote_code", True)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=kwargs.get("trust_remote_code", True)
        )
        if nla_num_layers is not None:
            needed = nla_num_layers + 1
            assert needed <= config.num_hidden_layers, (
                f"nla_num_layers={nla_num_layers} needs blocks 0..{nla_num_layers} "
                f"inclusive ({needed=}), but base model has {config.num_hidden_layers}."
            )
            _truncate_config_layers(config, needed)

        backbone = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=config, **kwargs,
        )
        # Strip lm_head — critic never produces logits
        if hasattr(backbone, "lm_head"):
            backbone.lm_head = nn.Identity()

        # Strip final layernorm — raw residual stream → value head
        inner = _inner_transformer(backbone)
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, nn.Identity())
                break

        model = cls(config, backbone)

        # Load saved value_head weights if resuming from checkpoint, else identity-init
        head_path = Path(pretrained_model_name_or_path) / "value_head.safetensors"
        if head_path.exists():
            from safetensors.torch import load_file
            model.value_head.load_state_dict(load_file(str(head_path)))
        else:
            with torch.no_grad():
                model.value_head.weight.copy_(torch.eye(config.hidden_size))

        # Ensure value_head is on the same device AND dtype as the backbone.
        # When using device_map, backbone params are auto-placed but value_head
        # (a custom nn.Linear) stays on CPU. Move it to match the first backbone param.
        backbone_param = next(model.backbone.parameters())
        model.value_head = model.value_head.to(
            device=backbone_param.device, dtype=backbone_param.dtype,
        )

        return model

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        inner = _inner_transformer(self.backbone)
        out = inner(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        h = out.last_hidden_state  # [B, T, d]
        return NLACriticOutput(values=self.value_head(h), backbone_last_hidden=h)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {k.removeprefix("backbone."): v for k, v in state_dict.items()
                       if k.startswith("backbone.")}
        head_sd = {k.removeprefix("value_head."): v for k, v in state_dict.items()
                   if k.startswith("value_head.")}
        self.backbone.save_pretrained(save_directory, state_dict=backbone_sd, **kwargs)
        save_file(head_sd, str(Path(save_directory) / "value_head.safetensors"))

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()
