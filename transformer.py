import yaml
import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder

from utils import (
    validate_block_configs, 
    validate_decoder_block_configs,
    create_causal_mask,
    make_repeated_block_configs,
    make_repeated_decoder_block_configs
)

from padding_masks import (
    create_encoder_padding_mask,
    create_cross_padding_mask,
    create_decoder_self_padding_mask,
    combine_masks
)


class Transformer(nn.Module):
    """
    Transformer encoder-decoder configurable.

    Soporta:
    - configuración explícita bloque a bloque
    - configuración por patrón repetido
    - creación desde dict en código
    - creación desde archivo YAML

    Supone que ya existen:
    - TransformerEncoder
    - TransformerDecoder
    - validate_block_configs
    - validate_decoder_block_configs
    - make_repeated_block_configs
    - make_repeated_decoder_block_configs
    - create_causal_mask
    - create_encoder_padding_mask
    - create_decoder_self_padding_mask
    - create_cross_padding_mask
    - combine_masks
    """

    def __init__(
        self,
        vocab_size_src: int,
        vocab_size_tgt: int,
        encoder_block_configs: list[dict],
        decoder_block_configs: list[dict],
        max_len_src: int = 5000,
        max_len_tgt: int = 5000,
        pad_idx_src: int = 0,
        pad_idx_tgt: int = 0,
        encoder_input_dropout: float = 0.0,
        decoder_input_dropout: float = 0.0,
        bias: bool = False,
        encoder_final_norm: bool = True,
        decoder_final_norm: bool = True,
        decoder_output_logits: bool = True,
    ):
        super().__init__()

        validate_block_configs(encoder_block_configs)
        validate_decoder_block_configs(decoder_block_configs)

        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt

        self.encoder_block_configs = encoder_block_configs
        self.decoder_block_configs = decoder_block_configs

        self.pad_idx_src = pad_idx_src
        self.pad_idx_tgt = pad_idx_tgt

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size_src,
            block_configs=encoder_block_configs,
            max_len=max_len_src,
            pad_idx=pad_idx_src,
            input_dropout=encoder_input_dropout,
            bias=bias,
            final_norm=encoder_final_norm,
        )

        encoder_dim = self.encoder.output_dim

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size_tgt,
            block_configs=decoder_block_configs,
            encoder_dim=encoder_dim,
            max_len=max_len_tgt,
            pad_idx=pad_idx_tgt,
            input_dropout=decoder_input_dropout,
            bias=bias,
            final_norm=decoder_final_norm,
            output_logits=decoder_output_logits,
        )

        self.encoder_output_dim = self.encoder.output_dim
        self.decoder_output_dim = self.decoder.output_dim
        self.decoder_output_logits = decoder_output_logits

    def forward(
        self,
        src_token_ids: torch.Tensor,
        tgt_token_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_self_mask: torch.Tensor | None = None,
        tgt_cross_mask: torch.Tensor | None = None,
        use_causal_tgt_mask: bool = True,
        auto_padding_masks: bool = True,
        return_attentions: bool = False,
    ):
        """
        src_token_ids: [B, T_src]
        tgt_token_ids: [B, T_tgt]
        """

        if auto_padding_masks:
            if src_mask is None:
                src_mask = create_encoder_padding_mask(
                    src_token_ids,
                    pad_idx_src=self.pad_idx_src,
                )

            if tgt_cross_mask is None:
                tgt_cross_mask = create_cross_padding_mask(
                    src_token_ids,
                    pad_idx_src=self.pad_idx_src,
                )

            if tgt_self_mask is None:
                tgt_pad_mask = create_decoder_self_padding_mask(
                    tgt_token_ids,
                    pad_idx_tgt=self.pad_idx_tgt,
                )

                if use_causal_tgt_mask:
                    tgt_causal_mask = create_causal_mask(
                        tgt_token_ids.size(1),
                        device=tgt_token_ids.device,
                    )
                    tgt_self_mask = combine_masks(tgt_pad_mask, tgt_causal_mask)
                else:
                    tgt_self_mask = tgt_pad_mask

        encoder_out = self.encoder(
            token_ids=src_token_ids,
            mask=src_mask,
            return_attentions=return_attentions,
        )

        encoder_attentions = None
        if return_attentions:
            encoder_out, encoder_attentions = encoder_out

        decoder_out = self.decoder(
            token_ids=tgt_token_ids,
            encoder_out=encoder_out,
            self_mask=tgt_self_mask,
            cross_mask=tgt_cross_mask,
            return_attentions=return_attentions,
        )

        decoder_attentions = None
        if return_attentions:
            decoder_out, decoder_attentions = decoder_out
            return {
                "output": decoder_out,
                "encoder_attentions": encoder_attentions,
                "decoder_attentions": decoder_attentions,
            }

        return decoder_out

    @classmethod
    def from_config(cls, config: dict) -> "Transformer":
        if not isinstance(config, dict):
            raise ValueError("config debe ser un dict")

        model_cfg = config.get("model", config)

        vocab_size_src = model_cfg["vocab_size_src"]
        vocab_size_tgt = model_cfg["vocab_size_tgt"]

        encoder_cfg = model_cfg["encoder"]
        decoder_cfg = model_cfg["decoder"]

        encoder_block_configs = cls._resolve_encoder_block_configs(encoder_cfg)
        decoder_block_configs = cls._resolve_decoder_block_configs(decoder_cfg)

        return cls(
            vocab_size_src=vocab_size_src,
            vocab_size_tgt=vocab_size_tgt,
            encoder_block_configs=encoder_block_configs,
            decoder_block_configs=decoder_block_configs,
            max_len_src=model_cfg.get("max_len_src", 5000),
            max_len_tgt=model_cfg.get("max_len_tgt", 5000),
            pad_idx_src=model_cfg.get("pad_idx_src", 0),
            pad_idx_tgt=model_cfg.get("pad_idx_tgt", 0),
            encoder_input_dropout=model_cfg.get("encoder_input_dropout", 0.0),
            decoder_input_dropout=model_cfg.get("decoder_input_dropout", 0.0),
            bias=model_cfg.get("bias", False),
            encoder_final_norm=model_cfg.get("encoder_final_norm", True),
            decoder_final_norm=model_cfg.get("decoder_final_norm", True),
            decoder_output_logits=model_cfg.get("decoder_output_logits", True),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Transformer":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls.from_config(config)

    @staticmethod
    def _resolve_encoder_block_configs(encoder_cfg: dict) -> list[dict]:
        if "block_configs" in encoder_cfg:
            block_configs = encoder_cfg["block_configs"]
            validate_block_configs(block_configs)
            return block_configs

        if "pattern" in encoder_cfg and "num_layers" in encoder_cfg:
            pattern = encoder_cfg["pattern"]
            num_layers = encoder_cfg["num_layers"]

            block_configs = make_repeated_block_configs(
                num_layers=num_layers,
                d_model=pattern["d_model"],
                num_heads=pattern["num_heads"],
                d_ff=pattern["d_ff"],
                dropout=pattern.get("dropout", 0.0),
                activation=pattern.get("activation", "gelu"),
                extra_fields=pattern.get("extra_fields"),
            )
            return block_configs

        raise ValueError(
            "Configuración inválida para encoder. "
            "Debes proveer 'block_configs' o bien 'pattern' + 'num_layers'."
        )

    @staticmethod
    def _resolve_decoder_block_configs(decoder_cfg: dict) -> list[dict]:
        if "block_configs" in decoder_cfg:
            block_configs = decoder_cfg["block_configs"]
            validate_decoder_block_configs(block_configs)
            return block_configs

        if "pattern" in decoder_cfg and "num_layers" in decoder_cfg:
            pattern = decoder_cfg["pattern"]
            num_layers = decoder_cfg["num_layers"]

            block_configs = make_repeated_decoder_block_configs(
                num_layers=num_layers,
                d_model=pattern["d_model"],
                num_heads=pattern["num_heads"],
                d_ff=pattern["d_ff"],
                dropout=pattern.get("dropout", 0.0),
                activation=pattern.get("activation", "gelu"),
                cross_num_heads=pattern.get("cross_num_heads"),
                extra_fields=pattern.get("extra_fields"),
            )
            return block_configs

        raise ValueError(
            "Configuración inválida para decoder. "
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

        lines.append("Transformer Summary")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        total_params = 0
        total_trainable = 0

        for name, module in self.named_modules():
            if name == "":
                continue

            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            module_trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            if module_params == 0:
                continue

            total_params += module_params
            total_trainable += module_trainable

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
        lines.append(
            f"{'Non-trainable params:':<30}{self._format_int(counts['non_trainable'])}"
        )
        lines.append(sep)

        lines.append("Encoder")
        lines.append(
            f"  blocks: {len(self.encoder_block_configs)}, "
            f"output_dim: {self.encoder_output_dim}"
        )

        lines.append("Decoder")
        lines.append(
            f"  blocks: {len(self.decoder_block_configs)}, "
            f"output_dim: {self.decoder_output_dim}, "
            f"output_logits: {self.decoder_output_logits}"
        )

        return "\n".join(lines)

    def print_summary(self, max_name_width: int = 60) -> None:
        print(self.summary(max_name_width=max_name_width))