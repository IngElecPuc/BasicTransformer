import torch
from copy import deepcopy


VALID_ACTIVATIONS = {"gelu", "relu"}


def _validate_common_block_config(
    cfg: dict,
    block_idx: int,
    require_cross_num_heads: bool = False,
) -> None:
    required_keys = {"d_model", "num_heads", "d_ff"}
    missing = required_keys - set(cfg.keys())
    if missing:
        raise ValueError(
            f"Al bloque {block_idx} le faltan claves requeridas: {missing}"
        )

    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    d_ff = cfg["d_ff"]
    dropout = cfg.get("dropout", 0.0)
    activation = cfg.get("activation", "gelu")

    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError(f"Bloque {block_idx}: d_model debe ser int > 0")

    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(f"Bloque {block_idx}: num_heads debe ser int > 0")

    if not isinstance(d_ff, int) or d_ff <= 0:
        raise ValueError(f"Bloque {block_idx}: d_ff debe ser int > 0")

    if d_model % num_heads != 0:
        raise ValueError(
            f"Bloque {block_idx}: d_model={d_model} no es divisible por num_heads={num_heads}"
        )

    if not isinstance(dropout, (int, float)) or not (0.0 <= float(dropout) < 1.0):
        raise ValueError(
            f"Bloque {block_idx}: dropout debe estar en el rango [0.0, 1.0)"
        )

    if not isinstance(activation, str) or activation.lower() not in VALID_ACTIVATIONS:
        raise ValueError(
            f"Bloque {block_idx}: activation debe ser una de {VALID_ACTIVATIONS}"
        )

    if require_cross_num_heads:
        cross_num_heads = cfg.get("cross_num_heads", num_heads)

        if not isinstance(cross_num_heads, int) or cross_num_heads <= 0:
            raise ValueError(
                f"Bloque {block_idx}: cross_num_heads debe ser int > 0"
            )

        if d_model % cross_num_heads != 0:
            raise ValueError(
                f"Bloque {block_idx}: d_model={d_model} no es divisible por "
                f"cross_num_heads={cross_num_heads}"
            )


def validate_block_configs(block_configs: list[dict]) -> None:
    if not isinstance(block_configs, list) or len(block_configs) == 0:
        raise ValueError("block_configs debe ser una lista no vacía")

    for i, cfg in enumerate(block_configs):
        if not isinstance(cfg, dict):
            raise ValueError(f"Bloque {i}: la configuración debe ser un dict")

        _validate_common_block_config(
            cfg=cfg,
            block_idx=i,
            require_cross_num_heads=False,
        )


def validate_decoder_block_configs(block_configs: list[dict]) -> None:
    if not isinstance(block_configs, list) or len(block_configs) == 0:
        raise ValueError("block_configs debe ser una lista no vacía")

    for i, cfg in enumerate(block_configs):
        if not isinstance(cfg, dict):
            raise ValueError(f"Bloque {i}: la configuración debe ser un dict")

        _validate_common_block_config(
            cfg=cfg,
            block_idx=i,
            require_cross_num_heads=True,
        )


def make_repeated_block_configs(
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    dropout: float = 0.0,
    activation: str = "gelu",
    extra_fields: dict | None = None,
) -> list[dict]:
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers debe ser int > 0")

    base_config = {
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "dropout": dropout,
        "activation": activation,
    }

    if extra_fields is not None:
        if not isinstance(extra_fields, dict):
            raise ValueError("extra_fields debe ser un dict o None")
        base_config.update(extra_fields)

    block_configs = [deepcopy(base_config) for _ in range(num_layers)]
    validate_block_configs(block_configs)
    return block_configs


def make_repeated_decoder_block_configs(
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    dropout: float = 0.0,
    activation: str = "gelu",
    cross_num_heads: int | None = None,
    extra_fields: dict | None = None,
) -> list[dict]:
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers debe ser int > 0")

    base_config = {
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "dropout": dropout,
        "activation": activation,
        "cross_num_heads": cross_num_heads if cross_num_heads is not None else num_heads,
    }

    if extra_fields is not None:
        if not isinstance(extra_fields, dict):
            raise ValueError("extra_fields debe ser un dict o None")
        base_config.update(extra_fields)

    block_configs = [deepcopy(base_config) for _ in range(num_layers)]
    validate_decoder_block_configs(block_configs)
    return block_configs


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Para un decoder autoregresivo, el token de la posición t no debe mirar posiciones futuras.
    returns:
        [1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)