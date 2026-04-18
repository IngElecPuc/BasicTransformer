import torch
import torch.nn as nn
from .bert_model import BertModel
from .bert_heads import BertPreTrainingHeads
from memory_estimator import estimate_bert_activations, build_memory_estimate


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

        lines.append(sep)
        lines.append(f"{'Total params:':<30}{self._format_int(counts['total'])}")
        lines.append(f"{'Trainable params:':<30}{self._format_int(counts['trainable'])}")
        lines.append(f"{'Non-trainable params:':<30}{self._format_int(counts['non_trainable'])}")
        lines.append(sep)

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
        include_attention_map: bool = True,
        misc_runtime_bytes: int = 0,
    ) -> dict:
        activations_bytes, details = estimate_bert_activations(
            block_configs=self.bert.block_configs,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=self.bert.vocab_size,
            hidden_size=self.bert.block_configs[0]["d_model"],
            dtype=activation_dtype,
            include_pooler=True,
            include_mlm_logits=True,
            include_nsp_logits=True,
            include_attention_map=include_attention_map,
        )

        details.update({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "quantization_bits": quantization_bits,
            "optimizer": optimizer,
            "activation_dtype": str(activation_dtype),
            "model_kind": "bert_pretraining",
        })

        return build_memory_estimate(
            self,
            activations_bytes=activations_bytes,
            activation_details=details,
            quantization_bits=quantization_bits,
            optimizer=optimizer,
            misc_runtime_bytes=misc_runtime_bytes,
        )