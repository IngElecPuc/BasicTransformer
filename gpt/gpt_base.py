import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import create_gpt_padding_mask, create_causal_mask, combine_masks
from transformer.transformer_block import TransformerBlock
from transformer.utils import validate_block_configs, make_repeated_block_configs

class GPTEmbeddings(nn.Module):
    """
    Embeddings tipo GPT:
    - token embeddings
    - position embeddings aprendidas
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 1024,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_idx = pad_idx

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx,
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        position_ids: [B, T]

        returns:
            [B, T, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.dropout(x)
        return x


class GPTDecoder(nn.Module):
    """
    Stack decoder-only, autoregresivo.
    Permite:
    - block_configs explícito
    - patrón repetido
    - transiciones entre bloques si cambia d_model
    """

    def __init__(
        self,
        block_configs: list[dict],
        bias: bool = False,
        final_norm: bool = True,
    ):
        super().__init__()

        validate_block_configs(block_configs)

        self.block_configs = block_configs
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        prev_d_model = block_configs[0]["d_model"]

        for i, cfg in enumerate(block_configs):
            d_model = cfg["d_model"]
            num_heads = cfg["num_heads"]
            d_ff = cfg["d_ff"]
            dropout = cfg.get("dropout", 0.0)
            activation = cfg.get("activation", "gelu")

            if i == 0:
                self.transitions.append(nn.Identity())
            else:
                if prev_d_model == d_model:
                    self.transitions.append(nn.Identity())
                else:
                    self.transitions.append(nn.Linear(prev_d_model, d_model, bias=bias))

            block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                bias=bias,
                activation=activation,
            )

            self.blocks.append(block)
            prev_d_model = d_model

        self.final_norm = nn.LayerNorm(prev_d_model) if final_norm else nn.Identity()
        self.output_dim = prev_d_model

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        x: [B, T, H]
        attention_mask: broadcastable a [B, T, T]
        """
        all_hidden_states = []
        all_attentions = []

        if return_hidden_states:
            all_hidden_states.append(x)

        for transition, block in zip(self.transitions, self.blocks):
            x = transition(x)

            if return_attentions:
                x, attn_weights = block(
                    x=x,
                    mask=attention_mask,
                    return_attention=True,
                )
                all_attentions.append(attn_weights)
            else:
                x = block(
                    x=x,
                    mask=attention_mask,
                    return_attention=False,
                )

            if return_hidden_states:
                all_hidden_states.append(x)

        x = self.final_norm(x)

        outputs = {"last_hidden_state": x}

        if return_hidden_states:
            outputs["hidden_states"] = all_hidden_states

        if return_attentions:
            outputs["attentions"] = all_attentions

        return outputs


class GPTModel(nn.Module):
    """
    Modelo base GPT (decoder-only).
    Soporta:
    - creación desde config en código
    - creación desde YAML
    - configuración repetida
    - configuración bloque a bloque
    """

    def __init__(
        self,
        vocab_size: int,
        block_configs: list[dict],
        max_position_embeddings: int = 1024,
        pad_idx: int = 0,
        embedding_dropout: float = 0.1,
        bias: bool = False,
        final_norm: bool = True,
    ):
        super().__init__()

        validate_block_configs(block_configs)

        self.vocab_size = vocab_size
        self.block_configs = block_configs
        self.pad_idx = pad_idx
        self.max_position_embeddings = max_position_embeddings

        hidden_size = block_configs[0]["d_model"]

        self.embeddings = GPTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            pad_idx=pad_idx,
            dropout=embedding_dropout,
        )

        self.decoder = GPTDecoder(
            block_configs=block_configs,
            bias=bias,
            final_norm=final_norm,
        )

        self.hidden_size = self.decoder.output_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        auto_causal_mask: bool = True,
        auto_padding_mask: bool = True,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        input_ids: [B, T]
        attention_mask: broadcastable a [B, T, T] o [B, 1, T]
        """
        if attention_mask is None:
            masks = []

            if auto_padding_mask:
                masks.append(create_gpt_padding_mask(input_ids, pad_idx=self.pad_idx))

            if auto_causal_mask:
                masks.append(create_causal_mask(input_ids.size(1), device=input_ids.device))

            attention_mask = None
            for m in masks:
                attention_mask = combine_masks(attention_mask, m)

        x = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        outputs = self.decoder(
            x=x,
            attention_mask=attention_mask,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )

        return outputs

    @classmethod
    def from_config(cls, config: dict) -> "GPTModel":
        if not isinstance(config, dict):
            raise ValueError("config debe ser un dict")

        model_cfg = config.get("model", config)

        if "gpt" in model_cfg:
            model_cfg = model_cfg["gpt"]

        block_configs = cls._resolve_block_configs(model_cfg)

        return cls(
            vocab_size=model_cfg["vocab_size"],
            block_configs=block_configs,
            max_position_embeddings=model_cfg.get("max_position_embeddings", 1024),
            pad_idx=model_cfg.get("pad_idx", 0),
            embedding_dropout=model_cfg.get("embedding_dropout", 0.1),
            bias=model_cfg.get("bias", False),
            final_norm=model_cfg.get("final_norm", True),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GPTModel":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)

    @staticmethod
    def _resolve_block_configs(model_cfg: dict) -> list[dict]:
        if "block_configs" in model_cfg:
            block_configs = model_cfg["block_configs"]
            validate_block_configs(block_configs)
            return block_configs

        if "pattern" in model_cfg and "num_layers" in model_cfg:
            pattern = model_cfg["pattern"]
            num_layers = model_cfg["num_layers"]

            return make_repeated_block_configs(
                num_layers=num_layers,
                d_model=pattern["d_model"],
                num_heads=pattern["num_heads"],
                d_ff=pattern["d_ff"],
                dropout=pattern.get("dropout", 0.0),
                activation=pattern.get("activation", "gelu"),
                extra_fields=pattern.get("extra_fields"),
            )

        raise ValueError(
            "Configuración inválida para GPT. "
            "Debes proveer 'block_configs' o bien 'pattern' + 'num_layers'."
        )

    @staticmethod
    def _format_int(n: int) -> str:
        return f"{n:,}"

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    def summary(self, max_name_width: int = 60) -> str:
        lines = []
        sep = "-" * 120

        header = (
            f"{'Layer (name)':<{max_name_width}}"
            f"{'Type':<25}"
            f"{'Params':>15}"
            f"{'Trainable':>15}"
        )

        lines.append("GPTModel Summary")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for name, module in self.named_modules():
            if name == "":
                continue

            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            module_trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            if module_params == 0:
                continue

            display_name = name
            if len(display_name) > max_name_width - 3:
                display_name = display_name[: max_name_width - 3] + "..."

            lines.append(
                f"{display_name:<{max_name_width}}"
                f"{module.__class__.__name__:<25}"
                f"{self._format_int(module_params):>15}"
                f"{self._format_int(module_trainable):>15}"
            )

        counts = self.parameter_counts()

        lines.append(sep)
        lines.append(f"{'Total params:':<30}{self._format_int(counts['total'])}")
        lines.append(f"{'Trainable params:':<30}{self._format_int(counts['trainable'])}")
        lines.append(f"{'Non-trainable params:':<30}{self._format_int(counts['non_trainable'])}")
        lines.append(sep)
        lines.append(f"Blocks: {len(self.block_configs)}")
        lines.append(f"Hidden size inicial: {self.block_configs[0]['d_model']}")
        lines.append(f"Hidden size final: {self.hidden_size}")

        return "\n".join(lines)

    def print_summary(self, max_name_width: int = 60) -> None:
        print(self.summary(max_name_width=max_name_width))


class GPTLMHeadModel(nn.Module):
    """
    GPT para next-token prediction.
    """

    def __init__(
        self,
        gpt: GPTModel,
        tie_word_embeddings: bool = True,
        lm_head_bias: bool = False,
    ):
        super().__init__()

        self.gpt = gpt
        self.lm_head = nn.Linear(
            self.gpt.hidden_size,
            self.gpt.vocab_size,
            bias=lm_head_bias,
        )

        if tie_word_embeddings:
            if self.gpt.hidden_size != self.gpt.embeddings.token_embeddings.embedding_dim:
                raise ValueError(
                    "No se pueden atar embeddings si hidden_size final "
                    "no coincide con el tamaño de embeddings de entrada"
                )
            self.lm_head.weight = self.gpt.embeddings.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        auto_causal_mask: bool = True,
        auto_padding_mask: bool = True,
        labels: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        outputs = self.gpt(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            auto_causal_mask=auto_causal_mask,
            auto_padding_mask=auto_padding_mask,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        result = {"logits": logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            result["loss"] = loss

        if return_hidden_states:
            result["hidden_states"] = outputs.get("hidden_states")

        if return_attentions:
            result["attentions"] = outputs.get("attentions")

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Generación autoregresiva simple.
        """
        self.eval()

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.gpt.max_position_embeddings:
                input_cond = input_ids[:, -self.gpt.max_position_embeddings :]
            else:
                input_cond = input_ids

            outputs = self(
                input_ids=input_cond,
                auto_causal_mask=True,
                auto_padding_mask=True,
                labels=None,
                return_attentions=False,
                return_hidden_states=False,
            )

            logits = outputs["logits"][:, -1, :]

            if temperature <= 0:
                raise ValueError("temperature debe ser > 0")

            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_values,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None:
                if torch.all(next_token.squeeze(-1) == eos_token_id):
                    break

        return input_ids

    @staticmethod
    def _format_int(n: int) -> str:
        return f"{n:,}"

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    def summary(self, max_name_width: int = 60) -> str:
        lines = []
        sep = "-" * 120

        header = (
            f"{'Layer (name)':<{max_name_width}}"
            f"{'Type':<25}"
            f"{'Params':>15}"
            f"{'Trainable':>15}"
        )

        lines.append("GPTLMHeadModel Summary")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for name, module in self.named_modules():
            if name == "":
                continue

            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            module_trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            if module_params == 0:
                continue

            display_name = name
            if len(display_name) > max_name_width - 3:
                display_name = display_name[: max_name_width - 3] + "..."

            lines.append(
                f"{display_name:<{max_name_width}}"
                f"{module.__class__.__name__:<25}"
                f"{self._format_int(module_params):>15}"
                f"{self._format_int(module_trainable):>15}"
            )

        counts = self.parameter_counts()

        lines.append(sep)
        lines.append(f"{'Total params:':<30}{self._format_int(counts['total'])}")
        lines.append(f"{'Trainable params:':<30}{self._format_int(counts['trainable'])}")
        lines.append(f"{'Non-trainable params:':<30}{self._format_int(counts['non_trainable'])}")
        lines.append(sep)

        return "\n".join(lines)

    def print_summary(self, max_name_width: int = 60) -> None:
        print(self.summary(max_name_width=max_name_width))

"""
gpt = GPTModel.from_config({
    "vocab_size": 32000,
    "max_position_embeddings": 1024,
    "pad_idx": 0,
    "embedding_dropout": 0.1,
    "bias": False,
    "final_norm": True,
    "num_layers": 8,
    "pattern": {
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "activation": "gelu",
    },
})

model = GPTLMHeadModel(gpt)

gpt = GPTModel.from_config({
    "vocab_size": 32000,
    "max_position_embeddings": 1024,
    "pad_idx": 0,
    "block_configs": [
        {"d_model": 256, "num_heads": 8, "d_ff": 1024, "dropout": 0.1, "activation": "gelu"},
        {"d_model": 256, "num_heads": 8, "d_ff": 1024, "dropout": 0.1, "activation": "gelu"},
        {"d_model": 384, "num_heads": 8, "d_ff": 1536, "dropout": 0.1, "activation": "gelu"},
        {"d_model": 384, "num_heads": 12, "d_ff": 1536, "dropout": 0.1, "activation": "relu"},
    ],
})

model = GPTLMHeadModel(gpt)

YAML
model:
  gpt:
    vocab_size: 32000
    max_position_embeddings: 1024
    pad_idx: 0
    embedding_dropout: 0.1
    bias: false
    final_norm: true

    num_layers: 8
    pattern:
      d_model: 256
      num_heads: 8
      d_ff: 1024
      dropout: 0.1
      activation: gelu

gpt = GPTModel.from_yaml("gpt_config.yaml")
model = GPTLMHeadModel(gpt)

model:
  gpt:
    vocab_size: 32000
    max_position_embeddings: 1024
    pad_idx: 0

    block_configs:
      - d_model: 256
        num_heads: 8
        d_ff: 1024
        dropout: 0.1
        activation: gelu
      - d_model: 256
        num_heads: 8
        d_ff: 1024
        dropout: 0.1
        activation: gelu
      - d_model: 384
        num_heads: 8
        d_ff: 1536
        dropout: 0.1
        activation: gelu
"""