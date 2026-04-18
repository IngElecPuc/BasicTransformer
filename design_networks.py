from transformer.transformer import Transformer
from bert.bert_model import BertModel
from bert.bert_pretraining import BertForPreTraining
from gpt.gpt_model import GPTModel
from gpt.gpt_lm_head import GPTLMHeadModel
from memory_estimator import pretty_vram

drops3_ = [0.1, 0.1, 0.2]
drops5_ = [0.1, 0.1, 0.2, 0.2, 0.25]
drops6_ = [0.1, 0.1, 0.1, 0.2, 0.2, 0.25]

encoder_block_configs = [
    #{"d_model": 64, "num_heads": 4, "d_ff": 128, "dropout": drops5_[i], "activation": "gelu"} for i in range(5)
    {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
    #{"d_model": 128, "num_heads": 4, "d_ff": 512, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
]

decoder_block_configs = [
    #{"d_model": 64, "num_heads": 4, "d_ff": 128, "dropout": drops5_[i], "activation": "gelu"} for i in range(5)
    {"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
    #{"d_model": 128, "num_heads": 4, "d_ff": 512, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
]

def Tf_config(
    vocab_size,
    d_model,
    num_heads,
    d_ff,
    n_layers,
    dropouts
):

    encoder_block_configs = [
        #{"d_model": 64, "num_heads": 4, "d_ff": 128, "dropout": drops5_[i], "activation": "gelu"} for i in range(5)
        #{"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
        #{"d_model": 128, "num_heads": 4, "d_ff": 512, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
        #{"d_model": 256, "num_heads": 8, "d_ff": 768, "dropout": drops6_[i], "activation": "gelu"} for i in range(6)
        {"d_model": d_model, "num_heads": num_heads, "d_ff": d_ff, "dropout": dropouts[i], "activation": "gelu"} for i in range(n_layers)
    ]

    decoder_block_configs = [
        #{"d_model": 64, "num_heads": 4, "d_ff": 128, "dropout": drops5_[i], "activation": "gelu"} for i in range(5)
        #{"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
        #{"d_model": 128, "num_heads": 4, "d_ff": 512, "dropout": drops3_[i], "activation": "gelu"} for i in range(3)
        {"d_model": d_model, "num_heads": num_heads, "d_ff": d_ff, "dropout": dropouts[i], "activation": "gelu"} for i in range(n_layers)
    ]

    return Transformer(
        vocab_size_src=vocab_size,
        vocab_size_tgt=vocab_size,
        encoder_block_configs=encoder_block_configs,
        decoder_block_configs=decoder_block_configs,
        max_len_src=5000,
        max_len_tgt=5000,
        pad_idx_src=0,
        pad_idx_tgt=0,
        encoder_input_dropout=0.2,
        decoder_input_dropout=0.2,
        bias=False,
        encoder_final_norm=True,
        decoder_final_norm=True,
        decoder_output_logits=True,
    )

def BERT_config(
    vocab_size,
    d_model,
    num_heads,
    d_ff,
    n_layers,
    dropouts
):     

    bert_config = {
        "model": {
            "bert": {
                "vocab_size": vocab_size,
                "max_position_embeddings": 5000,
                "type_vocab_size": 2,
                "pad_idx": 0,
                "embedding_dropout": 0.2,
                "bias": False,
                "final_norm": True,
                "block_configs": [
                    {
                        "d_model": d_model, 
                        "num_heads": num_heads, 
                        "d_ff": d_ff, 
                        "dropout": dropouts[i], 
                        "activation": "gelu"
                    } for i in range(n_layers)
                    #{"d_model": 96, "num_heads": 6, "d_ff": 384, "dropout": drops6_[i], "activation": "gelu"},
                    #{"d_model": 64, "num_heads": 4, "d_ff": 256, "dropout": drops6_[i], "activation": "gelu"},
                    #{"d_model": 128, "num_heads": 8, "d_ff": 256, "dropout": drops6_[i], "activation": "gelu"},
                ],
            }
        }
    }

    bert = BertModel.from_config(bert_config)
    return BertForPreTraining(bert)

def GPT_config(
    vocab_size,
    d_model,
    num_heads,
    d_ff,
    n_layers,
    dropouts
): 

    gpt_config = {
        "model": {
            "gpt": {
                "vocab_size": vocab_size,
                "max_position_embeddings": 5000,
                "pad_idx": 0,
                "embedding_dropout": 0.2,
                "bias": False,
                "final_norm": True,
                "block_configs": [
                    {
                        "d_model": d_model, 
                        "num_heads": num_heads, 
                        "d_ff": d_ff, 
                        "dropout": dropouts[i], 
                        "activation": "gelu"
                    } for i in range(n_layers)
                ],
            }
        }
    }

    gpt = GPTModel.from_config(gpt_config)
    return GPTLMHeadModel(gpt)

if __name__ == "__main__":
    
    #Modelo "Grande para 4 Gb" d_model = 512, batch_size = 3
    #Modelo "Decente para 4 Gb" d_model = 256, batch_size = 16
    separator = 80
    
    print('='*separator)
    print('Raw Transformer\n')

    vocab_size = 50000
    drops_ = [0.1, 0.1, 0.1, 0.2, 0.2, 0.25, 0.25]
    d_model = 256
    num_heads = 8
    d_ff = d_model * 3
    n_layers = 7
    batch_size = 16
    
    tf_model = Tf_config(
        vocab_size = vocab_size,
        d_model = d_model,
        num_heads =num_heads,
        d_ff = d_ff,
        n_layers = n_layers,
        dropouts = drops_
        )
    
    tf_model.print_summary(verbosity=False)
    print('\n'+pretty_vram(tf_model.vram_size(batch_size=batch_size, src_seq_len=d_model, tgt_seq_len=d_model, quantization_bits=8, optimizer="adamw")))

    print('='*separator)
    print('BERT Pretraining\n')

    vocab_size = 50000
    drops_ = [0.1]*5 + [0.2]*5 + [0.25]*4
    d_model = 512
    num_heads = 8
    d_ff = d_model * 3
    n_layers = 12
    batch_size = 7
    bert_model = BERT_config(
        vocab_size = vocab_size,
        d_model = d_model,
        num_heads =num_heads,
        d_ff = d_ff,
        n_layers = n_layers,
        dropouts = drops_
        )
    bert_model.print_summary(verbosity=False)
    # print(bert.vram_size(batch_size=1, seq_len=128, quantization_bits=8, optimizer="adamw"))
    print('\n'+pretty_vram(bert_model.vram_size(batch_size=batch_size, seq_len=d_model, quantization_bits=8, optimizer="adamw")))
    
    print('='*separator)
    print('GPT \n')

    vocab_size = 50000
    drops_ = [0.1]*5 + [0.2]*5 + [0.25]*4
    d_model = 512
    num_heads = 8
    d_ff = d_model * 3
    n_layers = 12
    batch_size = 7
    gpt_model = GPT_config(
        vocab_size = vocab_size,
        d_model = d_model,
        num_heads =num_heads,
        d_ff = d_ff,
        n_layers = n_layers,
        dropouts = drops_
        )
    gpt_model.print_summary(verbosity=False)

    # print(gpt.vram_size(batch_size=batch_size, seq_len=128, quantization_bits=8, optimizer="adamw"))
    print('\n'+pretty_vram(gpt_model.vram_size(batch_size=batch_size, seq_len=d_model, quantization_bits=8, optimizer="adamw")))

    print('='*separator)
