import yaml
import torch
import torch.nn as nn
from .gpt_embeddings import GPTEmbeddings
from .gpt_decoder import GPTDecoder
from .utils import create_gpt_padding_mask, create_causal_mask, combine_masks
from transformer.utils import validate_block_configs, make_repeated_block_configs
from memory_estimator import estimate_gpt_activations, build_memory_estimate


class GPTModel(nn.Module):
    """
    Modelo base GPT (decoder-only).
    Soporta:
    - creación desde config en código
    - creación desde YAML
    - configuración repetida
    - configuración bloque a bloque

    GPT suele ser mejor para:
        generación
        autocompletado
        scoring autoregresivo
        instruction following
        tareas reformulables como “dado este prompt, genera esto”
        reward / preference modeling

    Propuestas:
    - GPTForSequenceClassification
    - GPTForTokenClassification, Menos típico que en BERT, pero posible.
    - GPTForQuestionAnswering
    - GPTForMultipleChoice
    - GPTForRegression
    - GPTForRewardModeling, útil para RLHF o ranking de respuestas
    - GPTForPreferenceModeling, parecido al de arriba
    - GPTForRetrievalScoring, para rankear documentos o respuestas condicionadas por un prompt
    - GPTForConditionalGeneration
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

    def summary(self, max_name_width: int = 60, verbosity: bool = True) -> str:
        counts = self.parameter_counts()

        if not verbosity:
            lines = []
            lines.append(f"{'Total params:':<30}{self._format_int(counts['total'])}")
            lines.append(f"{'Trainable params:':<30}{self._format_int(counts['trainable'])}")
            lines.append(f"{'Non-trainable params:':<30}{self._format_int(counts['non_trainable'])}")
            return "\n".join(lines)

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

        lines.append(sep)
        lines.append(f"{'Total params:':<30}{self._format_int(counts['total'])}")
        lines.append(f"{'Trainable params:':<30}{self._format_int(counts['trainable'])}")
        lines.append(f"{'Non-trainable params:':<30}{self._format_int(counts['non_trainable'])}")
        lines.append(sep)
        lines.append(f"Blocks: {len(self.block_configs)}")
        lines.append(f"Hidden size inicial: {self.block_configs[0]['d_model']}")
        lines.append(f"Hidden size final: {self.hidden_size}")

        return "\n".join(lines)

    def print_summary(self, max_name_width: int = 60, verbosity: bool = True) -> None:
        print(self.summary(max_name_width=max_name_width, verbosity=verbosity))

    def vram_size(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        quantization_bits: int | None = None,
        optimizer: str | None = None,
        activation_dtype: torch.dtype = torch.float32,
        include_logits: bool = False,
        include_attention_map: bool = True,
        misc_runtime_bytes: int = 0,
    ) -> dict:
        activations_bytes, details = estimate_gpt_activations(
            block_configs=self.block_configs,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=self.vocab_size,
            hidden_size=self.block_configs[0]["d_model"],
            dtype=activation_dtype,
            include_logits=include_logits,
            include_attention_map=include_attention_map,
        )

        details.update({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "quantization_bits": quantization_bits,
            "optimizer": optimizer,
            "activation_dtype": str(activation_dtype),
            "model_kind": "gpt_base",
        })

        return build_memory_estimate(
            self,
            activations_bytes=activations_bytes,
            activation_details=details,
            quantization_bits=quantization_bits,
            optimizer=optimizer,
            misc_runtime_bytes=misc_runtime_bytes,
        )