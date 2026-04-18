import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


_DTYPE_BYTES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return _DTYPE_BYTES.get(dtype, 4)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _safe_tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _unique_parameters(module: nn.Module) -> list[nn.Parameter]:
    seen = set()
    unique = []

    for param in module.parameters():
        if param is None:
            continue

        key = (
            param.untyped_storage().data_ptr(),
            param.storage_offset(),
            tuple(param.size()),
            tuple(param.stride()),
            param.dtype,
            param.device.type,
            str(param.device),
        )
        if key not in seen:
            seen.add(key)
            unique.append(param)

    return unique


def _unique_buffers(module: nn.Module) -> list[torch.Tensor]:
    seen = set()
    unique = []

    for buf in module.buffers():
        if buf is None:
            continue

        key = (
            buf.untyped_storage().data_ptr(),
            buf.storage_offset(),
            tuple(buf.size()),
            tuple(buf.stride()),
            buf.dtype,
            buf.device.type,
            str(buf.device),
        )
        if key not in seen:
            seen.add(key)
            unique.append(buf)

    return unique


def _quantized_num_bytes(numel: int, quantization_bits: int | None, default_dtype: torch.dtype) -> int:
    if quantization_bits is None:
        return numel * _dtype_nbytes(default_dtype)

    if quantization_bits <= 0:
        raise ValueError("quantization_bits debe ser > 0")

    return math.ceil(numel * quantization_bits / 8)

def pretty_vram(report: dict) -> str:
    def pick(*keys, default="N/D"):
        for key in keys:
            if key in report:
                return report[key]
        return default

    lines = []
    lines.append("Memoria estimada del modelo")
    lines.append("-" * 72)

    rows = [
        ("Modelo en memoria (sin cuantización)", pick("model_parameters_human_fp", "model_parameters_human_base")),
        ("Modelo en memoria (cuantizado)", pick("model_parameters_human_quantized")),
        ("Gradientes", pick("gradients_human")),
        ("Estado del optimizador", pick("optimizer_state_human")),
        ("Buffers persistentes", pick("buffers_human")),
        ("Mapa de activación", pick("activations_human")),
        ("Otros tensores temporales", pick("misc_runtime_human", default="0.00 B")),
        ("Total estimado en entrenamiento (sin cuantización)", pick("total_training_fp_human", "total_training_base_human")),
        ("Total estimado en inferencia (sin cuantización)", pick("total_inference_fp_human", "total_inference_base_human")),
        ("Total estimado en entrenamiento (cuantizado)", pick("total_training_quantized_human")),
        ("Total estimado en inferencia (cuantizado)", pick("total_inference_quantized_human")),
    ]

    label_width = max(len(label) for label, _ in rows) + 2

    for label, value in rows:
        lines.append(f"{label:<{label_width}}{value}")

    details = report.get("details", {})
    if details:
        lines.append("")
        lines.append("Detalle del escenario estimado")
        lines.append("-" * 72)

        detail_name_map = {
            "model_kind": "Tipo de modelo",
            "batch_size": "Batch size",
            "src_seq_len": "Longitud secuencia fuente",
            "tgt_seq_len": "Longitud secuencia objetivo",
            "seq_len": "Longitud de secuencia",
            "quantization_bits": "Bits de cuantización",
            "optimizer": "Optimizador",
            "activation_dtype": "Tipo de dato de activaciones",
            "encoder_embedding_bytes": "Embeddings del encoder",
            "decoder_embedding_bytes": "Embeddings del decoder",
            "embeddings_bytes": "Embeddings",
            "encoder_blocks_bytes": "Bloques del encoder",
            "decoder_blocks_bytes": "Bloques del decoder",
            "encoder_bytes": "Encoder",
            "decoder_bytes": "Decoder",
            "encoder_output_bytes": "Salida del encoder",
            "last_hidden_state_bytes": "Estado oculto final",
            "logits_bytes": "Logits",
            "pooler_bytes": "Pooler",
            "mlm_logits_bytes": "Logits MLM",
            "nsp_logits_bytes": "Logits NSP",
        }

        rendered_details = []
        for key, value in details.items():
            label = detail_name_map.get(
                key,
                key.replace("_", " ").capitalize()
            )

            if isinstance(value, int) and key.endswith("_bytes"):
                rendered_value = _format_bytes(value)
            else:
                rendered_value = str(value)

            rendered_details.append((label, rendered_value))

        if rendered_details:
            detail_width = max(len(label) for label, _ in rendered_details) + 2
            for label, value in rendered_details:
                lines.append(f"{label:<{detail_width}}{value}")

    lines.append("")
    lines.append("Notas")
    lines.append("-" * 72)
    lines.append("• 'Sin cuantización' es el tamaño base del modelo tal como normalmente vive en PyTorch.")
    lines.append("• 'Cuantizado' aplica la cuantización solo a los pesos del modelo, no a gradientes ni activaciones.")
    lines.append("• Los valores en MiB/GiB son una versión legible del tamaño en bytes.")

    return "\n".join(lines)

@dataclass
class MemoryEstimate:
    model_parameters_bytes_fp: int
    model_parameters_bytes_quantized: int
    gradients_bytes: int
    optimizer_state_bytes: int
    buffers_bytes: int
    activations_bytes: int
    misc_runtime_bytes: int
    total_training_fp_bytes: int
    total_inference_fp_bytes: int
    total_training_quantized_bytes: int
    total_inference_quantized_bytes: int
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_parameters_bytes_fp": self.model_parameters_bytes_fp,
            "model_parameters_human_fp": _format_bytes(self.model_parameters_bytes_fp),
            "model_parameters_bytes_quantized": self.model_parameters_bytes_quantized,
            "model_parameters_human_quantized": _format_bytes(self.model_parameters_bytes_quantized),
            "gradients_bytes": self.gradients_bytes,
            "gradients_human": _format_bytes(self.gradients_bytes),
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "optimizer_state_human": _format_bytes(self.optimizer_state_bytes),
            "buffers_bytes": self.buffers_bytes,
            "buffers_human": _format_bytes(self.buffers_bytes),
            "activations_bytes": self.activations_bytes,
            "activations_human": _format_bytes(self.activations_bytes),
            "misc_runtime_bytes": self.misc_runtime_bytes,
            "misc_runtime_human": _format_bytes(self.misc_runtime_bytes),
            "total_training_fp_bytes": self.total_training_fp_bytes,
            "total_training_fp_human": _format_bytes(self.total_training_fp_bytes),
            "total_inference_fp_bytes": self.total_inference_fp_bytes,
            "total_inference_fp_human": _format_bytes(self.total_inference_fp_bytes),
            "total_training_quantized_bytes": self.total_training_quantized_bytes,
            "total_training_quantized_human": _format_bytes(self.total_training_quantized_bytes),
            "total_inference_quantized_bytes": self.total_inference_quantized_bytes,
            "total_inference_quantized_human": _format_bytes(self.total_inference_quantized_bytes),
            "details": self.details,
        }


def _parameter_and_buffer_bytes(module: nn.Module, quantization_bits: int | None) -> tuple[int, int, int]:
    params = _unique_parameters(module)
    buffers = _unique_buffers(module)

    param_bytes_fp = sum(p.numel() * p.element_size() for p in params)
    param_bytes_q = sum(_quantized_num_bytes(p.numel(), quantization_bits, p.dtype) for p in params)
    buffer_bytes = sum(_safe_tensor_nbytes(b) for b in buffers)

    return param_bytes_fp, param_bytes_q, buffer_bytes


def _optimizer_state_bytes(module: nn.Module, optimizer: str | None) -> int:
    trainable = [p for p in _unique_parameters(module) if p.requires_grad]
    if not trainable or optimizer is None or optimizer.lower() == "none":
        return 0

    total = 0
    name = optimizer.lower()

    for p in trainable:
        pbytes = p.numel() * p.element_size()

        if name == "sgd":
            total += 0
        elif name in {"sgd_momentum", "momentum"}:
            total += pbytes
        elif name in {"adam", "adamw"}:
            total += 2 * pbytes
        else:
            raise ValueError("optimizer debe ser one of: None, 'sgd', 'momentum', 'adam', 'adamw'")

    return total


def _gradient_bytes(module: nn.Module) -> int:
    trainable = [p for p in _unique_parameters(module) if p.requires_grad]
    return sum(p.numel() * p.element_size() for p in trainable)


def _attention_block_activation_bytes(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    dtype: torch.dtype,
    include_attention_map: bool = True,
) -> int:
    bpe = _dtype_nbytes(dtype)
    head_dim = d_model // num_heads

    total = 0

    # layernorm input/output aproximado
    total += 2 * batch_size * seq_len_q * d_model * bpe

    # Q, K, V por cabeza en esta implementación no fusionada
    total += 3 * batch_size * num_heads * seq_len_q * head_dim * bpe

    # scores + probs
    if include_attention_map:
        total += 2 * batch_size * num_heads * seq_len_q * seq_len_kv * bpe

    # salida heads concatenada + proyección
    total += 2 * batch_size * seq_len_q * d_model * bpe

    # FFN intermedio + salida
    total += batch_size * seq_len_q * d_ff * bpe
    total += batch_size * seq_len_q * d_model * bpe

    return total


def estimate_transformer_activations(
    *,
    encoder_block_configs: list[dict],
    decoder_block_configs: list[dict],
    batch_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    src_vocab_size: int,
    tgt_vocab_size: int,
    encoder_dtype: torch.dtype,
    decoder_dtype: torch.dtype,
    include_logits: bool = True,
    include_attention_map: bool = True,
) -> tuple[int, dict[str, int]]:
    bpe_enc = _dtype_nbytes(encoder_dtype)
    bpe_dec = _dtype_nbytes(decoder_dtype)

    total = 0
    details = {}

    enc_d0 = encoder_block_configs[0]["d_model"]
    dec_d0 = decoder_block_configs[0]["d_model"]

    encoder_embed = batch_size * src_seq_len * enc_d0 * bpe_enc
    decoder_embed = batch_size * tgt_seq_len * dec_d0 * bpe_dec

    total += encoder_embed + decoder_embed
    details["encoder_embedding_bytes"] = encoder_embed
    details["decoder_embedding_bytes"] = decoder_embed

    prev = enc_d0
    enc_blocks_total = 0
    for cfg in encoder_block_configs:
        d_model = cfg["d_model"]
        num_heads = cfg["num_heads"]
        d_ff = cfg["d_ff"]

        if prev != d_model:
            enc_blocks_total += batch_size * src_seq_len * d_model * bpe_enc

        enc_blocks_total += _attention_block_activation_bytes(
            batch_size=batch_size,
            seq_len_q=src_seq_len,
            seq_len_kv=src_seq_len,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dtype=encoder_dtype,
            include_attention_map=include_attention_map,
        )
        prev = d_model

    total += enc_blocks_total
    details["encoder_blocks_bytes"] = enc_blocks_total

    encoder_out_dim = encoder_block_configs[-1]["d_model"]
    encoder_out_bytes = batch_size * src_seq_len * encoder_out_dim * bpe_enc
    total += encoder_out_bytes
    details["encoder_output_bytes"] = encoder_out_bytes

    prev = dec_d0
    dec_blocks_total = 0
    for cfg in decoder_block_configs:
        d_model = cfg["d_model"]
        self_heads = cfg["num_heads"]
        cross_heads = cfg.get("cross_num_heads", self_heads)
        d_ff = cfg["d_ff"]

        if prev != d_model:
            dec_blocks_total += batch_size * tgt_seq_len * d_model * bpe_dec

        dec_blocks_total += _attention_block_activation_bytes(
            batch_size=batch_size,
            seq_len_q=tgt_seq_len,
            seq_len_kv=tgt_seq_len,
            d_model=d_model,
            num_heads=self_heads,
            d_ff=d_ff,
            dtype=decoder_dtype,
            include_attention_map=include_attention_map,
        )

        dec_blocks_total += _attention_block_activation_bytes(
            batch_size=batch_size,
            seq_len_q=tgt_seq_len,
            seq_len_kv=src_seq_len,
            d_model=d_model,
            num_heads=cross_heads,
            d_ff=d_ff,
            dtype=decoder_dtype,
            include_attention_map=include_attention_map,
        )
        prev = d_model

    total += dec_blocks_total
    details["decoder_blocks_bytes"] = dec_blocks_total

    if include_logits:
        logits_bytes = batch_size * tgt_seq_len * tgt_vocab_size * bpe_dec
        total += logits_bytes
        details["logits_bytes"] = logits_bytes
    else:
        details["logits_bytes"] = 0

    return total, details


def estimate_bert_activations(
    *,
    block_configs: list[dict],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    include_pooler: bool = True,
    include_mlm_logits: bool = False,
    include_nsp_logits: bool = False,
    include_attention_map: bool = True,
) -> tuple[int, dict[str, int]]:
    bpe = _dtype_nbytes(dtype)
    total = 0
    details = {}

    embeddings_bytes = batch_size * seq_len * hidden_size * bpe
    total += embeddings_bytes
    details["embeddings_bytes"] = embeddings_bytes

    prev = block_configs[0]["d_model"]
    encoder_bytes = 0
    for cfg in block_configs:
        d_model = cfg["d_model"]
        num_heads = cfg["num_heads"]
        d_ff = cfg["d_ff"]

        if prev != d_model:
            encoder_bytes += batch_size * seq_len * d_model * bpe

        encoder_bytes += _attention_block_activation_bytes(
            batch_size=batch_size,
            seq_len_q=seq_len,
            seq_len_kv=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dtype=dtype,
            include_attention_map=include_attention_map,
        )
        prev = d_model

    total += encoder_bytes
    details["encoder_bytes"] = encoder_bytes

    last_hidden = batch_size * seq_len * block_configs[-1]["d_model"] * bpe
    total += last_hidden
    details["last_hidden_state_bytes"] = last_hidden

    if include_pooler:
        pooler_bytes = batch_size * block_configs[-1]["d_model"] * bpe
        total += pooler_bytes
        details["pooler_bytes"] = pooler_bytes
    else:
        details["pooler_bytes"] = 0

    if include_mlm_logits:
        mlm_logits = batch_size * seq_len * vocab_size * bpe
        total += mlm_logits
        details["mlm_logits_bytes"] = mlm_logits
    else:
        details["mlm_logits_bytes"] = 0

    if include_nsp_logits:
        nsp_logits = batch_size * 2 * bpe
        total += nsp_logits
        details["nsp_logits_bytes"] = nsp_logits
    else:
        details["nsp_logits_bytes"] = 0

    return total, details


def estimate_gpt_activations(
    *,
    block_configs: list[dict],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    include_logits: bool = True,
    include_attention_map: bool = True,
) -> tuple[int, dict[str, int]]:
    bpe = _dtype_nbytes(dtype)
    total = 0
    details = {}

    embeddings_bytes = batch_size * seq_len * hidden_size * bpe
    total += embeddings_bytes
    details["embeddings_bytes"] = embeddings_bytes

    prev = block_configs[0]["d_model"]
    decoder_bytes = 0
    for cfg in block_configs:
        d_model = cfg["d_model"]
        num_heads = cfg["num_heads"]
        d_ff = cfg["d_ff"]

        if prev != d_model:
            decoder_bytes += batch_size * seq_len * d_model * bpe

        decoder_bytes += _attention_block_activation_bytes(
            batch_size=batch_size,
            seq_len_q=seq_len,
            seq_len_kv=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dtype=dtype,
            include_attention_map=include_attention_map,
        )
        prev = d_model

    total += decoder_bytes
    details["decoder_bytes"] = decoder_bytes

    last_hidden = batch_size * seq_len * block_configs[-1]["d_model"] * bpe
    total += last_hidden
    details["last_hidden_state_bytes"] = last_hidden

    if include_logits:
        logits_bytes = batch_size * seq_len * vocab_size * bpe
        total += logits_bytes
        details["logits_bytes"] = logits_bytes
    else:
        details["logits_bytes"] = 0

    return total, details


def build_memory_estimate(
    module: nn.Module,
    *,
    activations_bytes: int,
    activation_details: dict[str, int],
    quantization_bits: int | None = None,
    optimizer: str | None = None,
    misc_runtime_bytes: int = 0,
) -> dict[str, Any]:
    model_fp, model_q, buffers = _parameter_and_buffer_bytes(module, quantization_bits)
    grads = _gradient_bytes(module)
    opt = _optimizer_state_bytes(module, optimizer)

    total_training_fp = model_fp + buffers + activations_bytes + grads + opt + misc_runtime_bytes
    total_inference_fp = model_fp + buffers + activations_bytes + misc_runtime_bytes

    total_training_q = model_q + buffers + activations_bytes + grads + opt + misc_runtime_bytes
    total_inference_q = model_q + buffers + activations_bytes + misc_runtime_bytes

    estimate = MemoryEstimate(
        model_parameters_bytes_fp=model_fp,
        model_parameters_bytes_quantized=model_q,
        gradients_bytes=grads,
        optimizer_state_bytes=opt,
        buffers_bytes=buffers,
        activations_bytes=activations_bytes,
        misc_runtime_bytes=misc_runtime_bytes,
        total_training_fp_bytes=total_training_fp,
        total_inference_fp_bytes=total_inference_fp,
        total_training_quantized_bytes=total_training_q,
        total_inference_quantized_bytes=total_inference_q,
        details=activation_details,
    )
    return estimate.as_dict()