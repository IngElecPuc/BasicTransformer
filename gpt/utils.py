import torch

def create_gpt_padding_mask(
    input_ids: torch.Tensor,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Máscara booleana de padding para GPT.

    input_ids: [B, T]

    returns:
        [B, 1, T]
    """
    return (input_ids != pad_idx).unsqueeze(1)


def create_causal_mask(
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Máscara causal booleana.

    returns:
        [1, T, T]
    """
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    ).unsqueeze(0)


def combine_masks(
    mask_a: torch.Tensor | None,
    mask_b: torch.Tensor | None,
) -> torch.Tensor | None:
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a
    return mask_a & mask_b