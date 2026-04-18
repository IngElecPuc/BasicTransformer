import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt_model import GPTModel


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

        lines.append(sep)
        lines.append(f"{'Total params:':<30}{self._format_int(counts['total'])}")
        lines.append(f"{'Trainable params:':<30}{self._format_int(counts['trainable'])}")
        lines.append(f"{'Non-trainable params:':<30}{self._format_int(counts['non_trainable'])}")
        lines.append(sep)

        return "\n".join(lines)

    def print_summary(self, max_name_width: int = 60, verbosity: bool = True) -> None:
        print(self.summary(max_name_width=max_name_width, verbosity=verbosity))


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