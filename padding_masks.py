import torch


def create_padding_mask(token_ids: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Crea una máscara booleana donde True indica una posición válida
    y False indica padding.

    token_ids: [B, T]

    returns:
        [B, 1, T]
    """
    return (token_ids != pad_idx).unsqueeze(1)


def create_encoder_padding_mask(
    src_token_ids: torch.Tensor,
    pad_idx_src: int = 0,
) -> torch.Tensor:
    """
    Máscara para self-attention del encoder.

    src_token_ids: [B, T_src]

    returns:
        [B, 1, T_src]
    """
    return create_padding_mask(src_token_ids, pad_idx=pad_idx_src)


def create_decoder_self_padding_mask(
    tgt_token_ids: torch.Tensor,
    pad_idx_tgt: int = 0,
) -> torch.Tensor:
    """
    Máscara de padding para self-attention del decoder.

    tgt_token_ids: [B, T_tgt]

    returns:
        [B, 1, T_tgt]
    """
    return create_padding_mask(tgt_token_ids, pad_idx=pad_idx_tgt)


def create_cross_padding_mask(
    src_token_ids: torch.Tensor,
    pad_idx_src: int = 0,
) -> torch.Tensor:
    """
    Máscara de padding para cross-attention del decoder.
    Enmascara las keys/values provenientes del encoder.

    src_token_ids: [B, T_src]

    returns:
        [B, 1, T_src]
    """
    return create_padding_mask(src_token_ids, pad_idx=pad_idx_src)


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Máscara causal para decoder self-attention.

    returns:
        [1, T, T]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0)


def combine_masks(
    mask_a: torch.Tensor | None,
    mask_b: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    Combina dos máscaras booleanas mediante AND lógico.
    Ambas deben ser broadcastables entre sí.

    returns:
        máscara combinada o None
    """
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a
    return mask_a & mask_b
