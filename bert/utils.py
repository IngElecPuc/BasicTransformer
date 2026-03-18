import torch

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