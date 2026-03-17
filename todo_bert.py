import yaml
import torch
import torch.nn as nn


def create_bert_padding_mask(
    input_ids: torch.Tensor,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Máscara booleana para self-attention bidireccional en BERT.

    input_ids: [B, T]

    returns:
        [B, 1, T]
    """
    return (input_ids != pad_idx).unsqueeze(1)


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


class BertPredictionHeadTransform(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size, bias=bias)

        if hidden_act.lower() == "gelu":
            self.transform_act_fn = nn.GELU()
        elif hidden_act.lower() == "relu":
            self.transform_act_fn = nn.ReLU()
        else:
            raise ValueError("hidden_act debe ser 'gelu' o 'relu'")

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    Head de MLM.
    Puede atar pesos con word_embeddings del modelo.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.transform = BertPredictionHeadTransform(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        self.decoder.weight = embedding_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        self.predictions.tie_weights(embedding_weight)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.predictions(sequence_output)


class BertOnlyNSPHead(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2, bias=bias)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return self.seq_relationship(pooled_output)


class BertPreTrainingHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.predictions = BertLMPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )
        self.seq_relationship = nn.Linear(hidden_size, 2, bias=bias)

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        self.predictions.tie_weights(embedding_weight)

    def forward(
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


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


class BertForMaskedLM(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.bert = bert
        self.cls = BertOnlyMLMHead(
            hidden_size=bert.hidden_size,
            vocab_size=bert.vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

        if tie_word_embeddings:
            self.cls.tie_weights(self.bert.embeddings.word_embeddings.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        auto_padding_mask: bool = True,
        labels: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            auto_padding_mask=auto_padding_mask,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )

        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)

        result = {"logits": prediction_scores}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
            )
            result["loss"] = loss

        if return_hidden_states:
            result["hidden_states"] = outputs.get("hidden_states")

        if return_attentions:
            result["attentions"] = outputs.get("attentions")

        return result


class BertForNextSentencePrediction(nn.Module):
    def __init__(self, bert: BertModel, bias: bool = True):
        super().__init__()
        self.bert = bert

        if self.bert.pooler is None:
            raise ValueError("BertModel debe haberse creado con add_pooling_layer=True")

        self.cls = BertOnlyNSPHead(hidden_size=bert.hidden_size, bias=bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        auto_padding_mask: bool = True,
        labels: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            auto_padding_mask=auto_padding_mask,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )

        pooled_output = outputs["pooler_output"]
        seq_relationship_scores = self.cls(pooled_output)

        result = {"logits": seq_relationship_scores}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                seq_relationship_scores.view(-1, 2),
                labels.view(-1),
            )
            result["loss"] = loss

        if return_hidden_states:
            result["hidden_states"] = outputs.get("hidden_states")

        if return_attentions:
            result["attentions"] = outputs.get("attentions")

        return result


class BertForPreTraining(nn.Module):
    """
    Pretraining estilo BERT original:
    - Masked Language Modeling
    - Next Sentence Prediction
    """

    def __init__(
        self,
        bert: BertModel,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.bert = bert

        if self.bert.pooler is None:
            raise ValueError("BertModel debe haberse creado con add_pooling_layer=True")

        self.cls = BertPreTrainingHeads(
            hidden_size=bert.hidden_size,
            vocab_size=bert.vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

        if tie_word_embeddings:
            self.cls.tie_weights(self.bert.embeddings.word_embeddings.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        auto_padding_mask: bool = True,
        mlm_labels: torch.Tensor | None = None,
        nsp_labels: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            auto_padding_mask=auto_padding_mask,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]

        prediction_scores, seq_relationship_scores = self.cls(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
        )

        result = {
            "prediction_logits": prediction_scores,
            "seq_relationship_logits": seq_relationship_scores,
        }

        total_loss = None

        if mlm_labels is not None:
            mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                mlm_labels.view(-1),
            )
            result["mlm_loss"] = mlm_loss
            total_loss = mlm_loss if total_loss is None else total_loss + mlm_loss

        if nsp_labels is not None:
            nsp_loss_fct = nn.CrossEntropyLoss()
            nsp_loss = nsp_loss_fct(
                seq_relationship_scores.view(-1, 2),
                nsp_labels.view(-1),
            )
            result["nsp_loss"] = nsp_loss
            total_loss = nsp_loss if total_loss is None else total_loss + nsp_loss

        if total_loss is not None:
            result["loss"] = total_loss

        if return_hidden_states:
            result["hidden_states"] = outputs.get("hidden_states")

        if return_attentions:
            result["attentions"] = outputs.get("attentions")

        return result

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    @staticmethod
    def _format_int(n: int) -> str:
        return f"{n:,}"

    def summary(self, max_name_width: int = 60) -> str:
        lines = []
        sep = "-" * 120

        header = (
            f"{'Layer (name)':<{max_name_width}}"
            f"{'Type':<25}"
            f"{'Params':>15}"
            f"{'Trainable':>15}"
        )

        lines.append("BertForPreTraining Summary")
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