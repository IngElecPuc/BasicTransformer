from transformer.transformer import Transformer

encoder_block_configs = [
    {
        "d_model": 64,
        "num_heads": 4,
        "d_ff": 128,
        "dropout": 0.1,
        "activation": "gelu",
    },
    {
        "d_model": 96,
        "num_heads": 6,
        "d_ff": 192,
        "dropout": 0.1,
        "activation": "relu",
    },
    {
        "d_model": 128,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.2,
        "activation": "gelu",
    },
]

decoder_block_configs = [
    {
        "d_model": 128,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.2,          # opcional
        "activation": "gelu",      # opcional
        "cross_num_heads": 8,    # opcional
    },
    {
        "d_model": 96,
        "num_heads": 6,
        "d_ff": 192,
        "dropout": 0.1,          
        "activation": "gelu",      
        "cross_num_heads": 6,    
    },
    {
        "d_model": 64,
        "num_heads": 4,
        "d_ff": 128,
        "dropout": 0.1,          
        "activation": "gelu",      
        "cross_num_heads": 4,    
    },

]


if __name__ == "__main__":
    
    
    model = Transformer(
        vocab_size_src = 5000,
        vocab_size_tgt = 5000,
        encoder_block_configs = encoder_block_configs,
        decoder_block_configs = decoder_block_configs,
        max_len_src = 5000,
        max_len_tgt = 5000,
        pad_idx_src = 0,
        pad_idx_tgt = 0,
        encoder_input_dropout = 0.2,
        decoder_input_dropout = 0.2,
        bias = False,
        encoder_final_norm = True,
        decoder_final_norm = True,
        decoder_output_logits = True,
    )

    model.print_summary()