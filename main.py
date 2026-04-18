from transformer.transformer import Transformer
from bert.bert_model import BertModel
from bert.bert_pretraining import BertForPreTraining
from gpt.gpt_model import GPTModel
from gpt.gpt_lm_head import GPTLMHeadModel

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

bert_config = {
    "model": {
        "bert": {
            "vocab_size": 5000,
            "max_position_embeddings": 5000,
            "type_vocab_size": 2,
            "pad_idx": 0,
            "embedding_dropout": 0.2,
            "bias": False,
            "final_norm": True,
            "block_configs": [
                {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": 0.1, "activation": "gelu"},
                {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": 0.1, "activation": "gelu"},
                {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": 0.1, "activation": "relu"},
                {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": 0.1, "activation": "gelu"},
                {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": 0.2, "activation": "gelu"},
                {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": 0.2, "activation": "gelu"},
                #{"d_model": 64, "num_heads": 4, "d_ff": 256, "dropout": 0.2, "activation": "gelu"},
            ],
        }
    }
}

gpt_config = {
    "model": {
        "gpt": {
            "vocab_size": 5000,
            "max_position_embeddings": 5000,
            "pad_idx": 0,
            "embedding_dropout": 0.2,
            "bias": False,
            "final_norm": True,
            "block_configs": [
                {"d_model": 128, "num_heads": 8, "d_ff": 256, "dropout": 0.1, "activation": "gelu"},
                {"d_model": 128, "num_heads": 8, "d_ff": 256, "dropout": 0.1, "activation": "gelu"},
                {"d_model": 128, "num_heads": 8, "d_ff": 256, "dropout": 0.1, "activation": "relu"},
                {"d_model": 128, "num_heads": 8, "d_ff": 256, "dropout": 0.1, "activation": "gelu"},
                {"d_model": 128, "num_heads": 8, "d_ff": 256, "dropout": 0.2, "activation": "gelu"},
                {"d_model": 128, "num_heads": 8, "d_ff": 256, "dropout": 0.2, "activation": "gelu"},
                #{"d_model": 64, "num_heads": 4, "d_ff": 256, "dropout": 0.2, "activation": "gelu"},
            ],
        }
    }
}

if __name__ == "__main__":
    
    
    tf_model = Transformer(
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

    bert = BertModel.from_config(bert_config)
    bert_model = BertForPreTraining(bert)

    gpt = GPTModel.from_config(gpt_config)
    gpt_model = GPTLMHeadModel(gpt)
    
    tf_model.print_summary(verbosity=False)
    bert_model.print_summary(verbosity=False)
    gpt_model.print_summary(verbosity=False)