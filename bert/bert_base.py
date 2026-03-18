import torch
import torch.nn as nn
from transformer.transformer_block import TransformerBlock
from transformer.utils import validate_block_configs, make_repeated_block_configs
from utils import create_bert_padding_mask
import yaml

class BertEmbeddings(nn.Module):
    """
    Embeddings de entrada estilo BERT:
    - token embeddings
    - position embeddings aprendidas
    - token type embeddings
    - LayerNorm
    - Dropout
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        pad_idx: int = 0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size

        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx,
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=hidden_size,
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=type_vocab_size,
            embedding_dim=hidden_size,
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids:      [B, T]
        token_type_ids: [B, T]
        position_ids:   [B, T]

        returns:
            [B, T, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertEncoder(nn.Module):
    """
    Encoder estilo BERT basado en Transformer blocks.
    Permite:
    - configuración repetitiva
    - configuración bloque a bloque
    - transiciones de dimensión entre bloques si cambian los d_model
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
    ):
        """
        x: [B, T, hidden_size_in]
        attention_mask: broadcastable a [B, T, T]

        returns:
            sequence_output: [B, T, hidden_size_out]
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

class BertPooler(nn.Module):
    """
    Pooler estilo BERT:
    toma el hidden state del token [CLS] y lo proyecta.
    """

    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]

        returns:
            [B, H]
        """
        cls_token_state = hidden_states[:, 0]
        pooled_output = self.dense(cls_token_state)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(nn.Module):
    """
    Modelo base BERT:
    - embeddings
    - encoder bidireccional
    - pooler

    Soporta construcción desde:
    - config en código
    - YAML
    """

    def __init__(
        self,
        vocab_size: int,
        block_configs: list[dict],
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        pad_idx: int = 0,
        embedding_dropout: float = 0.1,
        embedding_layer_norm_eps: float = 1e-5,
        bias: bool = False,
        final_norm: bool = True,
        add_pooling_layer: bool = True,
    ):
        super().__init__()

        validate_block_configs(block_configs)

        self.vocab_size = vocab_size
        self.block_configs = block_configs
        self.pad_idx = pad_idx
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.add_pooling_layer = add_pooling_layer

        hidden_size = block_configs[0]["d_model"]

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            pad_idx=pad_idx,
            dropout=embedding_dropout,
            layer_norm_eps=embedding_layer_norm_eps,
        )

        self.encoder = BertEncoder(
            block_configs=block_configs,
            bias=bias,
            final_norm=final_norm,
        )

        self.hidden_size = self.encoder.output_dim

        self.pooler = BertPooler(self.hidden_size, bias=True) if add_pooling_layer else None

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        auto_padding_mask: bool = True,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        input_ids:      [B, T]
        token_type_ids: [B, T]
        position_ids:   [B, T]
        attention_mask: broadcastable a [B, T, T] o [B, 1, T]
        """
        if auto_padding_mask and attention_mask is None:
            attention_mask = create_bert_padding_mask(
                input_ids=input_ids,
                pad_idx=self.pad_idx,
            )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        encoder_outputs = self.encoder(
            x=embedding_output,
            attention_mask=attention_mask,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )

        sequence_output = encoder_outputs["last_hidden_state"]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        outputs = {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
        }

        if return_hidden_states:
            outputs["hidden_states"] = encoder_outputs["hidden_states"]

        if return_attentions:
            outputs["attentions"] = encoder_outputs["attentions"]

        return outputs

    @classmethod
    def from_config(cls, config: dict) -> "BertModel":
        if not isinstance(config, dict):
            raise ValueError("config debe ser un dict")

        model_cfg = config.get("model", config)

        if "bert" in model_cfg:
            model_cfg = model_cfg["bert"]

        block_configs = cls._resolve_block_configs(model_cfg)

        return cls(
            vocab_size=model_cfg["vocab_size"],
            block_configs=block_configs,
            max_position_embeddings=model_cfg.get("max_position_embeddings", 512),
            type_vocab_size=model_cfg.get("type_vocab_size", 2),
            pad_idx=model_cfg.get("pad_idx", 0),
            embedding_dropout=model_cfg.get("embedding_dropout", 0.1),
            embedding_layer_norm_eps=model_cfg.get("embedding_layer_norm_eps", 1e-5),
            bias=model_cfg.get("bias", False),
            final_norm=model_cfg.get("final_norm", True),
            add_pooling_layer=model_cfg.get("add_pooling_layer", True),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BertModel":
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
            "Configuración inválida para BERT. "
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

        lines.append("BertModel Summary")
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

"""
bert = BertModel.from_config({
    "vocab_size": 30000,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "pad_idx": 0,
    "embedding_dropout": 0.1,
    "bias": False,
    "final_norm": True,
    "add_pooling_layer": True,
    "num_layers": 6,
    "pattern": {
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "activation": "gelu",
    },
})


bert = BertModel.from_config({
    "vocab_size": 30000,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "pad_idx": 0,
    "block_configs": [
        {"d_model": 256, "num_heads": 8, "d_ff": 1024, "dropout": 0.1, "activation": "gelu"},
        {"d_model": 256, "num_heads": 8, "d_ff": 1024, "dropout": 0.1, "activation": "gelu"},
        {"d_model": 384, "num_heads": 8, "d_ff": 1536, "dropout": 0.1, "activation": "gelu"},
    ],
})

bert = BertModel.from_config({
    "vocab_size": 30000,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "pad_idx": 0,
    "block_configs": [
        {"d_model": 256, "num_heads": 8, "d_ff": 1024, "dropout": 0.1, "activation": "gelu"},
        {"d_model": 256, "num_heads": 8, "d_ff": 1024, "dropout": 0.1, "activation": "gelu"},
        {"d_model": 384, "num_heads": 8, "d_ff": 1536, "dropout": 0.1, "activation": "gelu"},
    ],
})

input_ids = torch.randint(0, 30000, (2, 16))
token_type_ids = torch.zeros_like(input_ids)

outputs = model(
    input_ids=input_ids,
    token_type_ids=token_type_ids,
)

print(outputs["prediction_logits"].shape)       # [B, T, vocab_size]
print(outputs["seq_relationship_logits"].shape) # [B, 2]

YAML

model:
  bert:
    vocab_size: 30000
    max_position_embeddings: 512
    type_vocab_size: 2
    pad_idx: 0
    embedding_dropout: 0.1
    bias: false
    final_norm: true
    add_pooling_layer: true

    num_layers: 6
    pattern:
      d_model: 256
      num_heads: 8
      d_ff: 1024
      dropout: 0.1
      activation: gelu

model:
  bert:
    vocab_size: 30000
    max_position_embeddings: 512
    type_vocab_size: 2
    pad_idx: 0
    embedding_dropout: 0.1
    bias: false
    final_norm: true
    add_pooling_layer: true

    num_layers: 6
    pattern:
      d_model: 256
      num_heads: 8
      d_ff: 1024
      dropout: 0.1
      activation: gelu

model:
  bert:
    vocab_size: 30000
    max_position_embeddings: 512
    type_vocab_size: 2
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