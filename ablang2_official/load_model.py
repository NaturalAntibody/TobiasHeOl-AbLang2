import json, argparse
from pathlib import Path
import torch


ablang1_models = ["ablang1-heavy", "ablang1-light"]
ablang2_models = ["ablang2-paired"]


def load_model(local_model_dir: Path, device: str):

    if local_model_dir.name in ablang1_models:
        AbLang, tokenizer, hparams = fetch_ablang1(
            local_model_dir,
            device=device
        )
    elif local_model_dir.name in ablang2_models:
        AbLang, tokenizer, hparams = fetch_ablang2(
            local_model_dir,
            device=device
        )
    else: 
        assert False, f"Provided path, \"{local_model_dir}\", is not a valid model folder."   

    return AbLang, tokenizer, hparams
    
        
def fetch_ablang1(local_model_dir: Path, device: str):
    
    from .models.ablang1 import model as ablang_1_model
    from .models.ablang1 import tokenizers as ablang_1_tokenizer
    
    with open(local_model_dir / 'hparams.json', 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**json.load(f))    

    AbLang = ablang_1_model.AbLang(hparams)
    AbLang.load_state_dict(
        torch.load(
            local_model_dir / 'amodel.pt',
            map_location=torch.device(device)
        )
    )
    tokenizer = ablang_1_tokenizer.ABtokenizer(local_model_dir / 'vocab.json')
        
    return AbLang, tokenizer, hparams


def fetch_ablang2(local_model_dir: Path, device: str):
    
    from .models.ablang2 import ablang
    from .models.ablang2 import tokenizers
    
    with open(local_model_dir / 'hparams.json', 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**json.load(f))    
        
    AbLang = ablang.AbLang(
        vocab_size = hparams.vocab_size,
        hidden_embed_size = hparams.hidden_embed_size,
        n_attn_heads = hparams.n_attn_heads,
        n_encoder_blocks = hparams.n_encoder_blocks,
        padding_tkn = hparams.pad_tkn,
        mask_tkn = hparams.mask_tkn,
        layer_norm_eps = hparams.layer_norm_eps,
        a_fn = hparams.a_fn,
    )

    AbLang.load_state_dict(
        torch.load(
            local_model_dir / 'model.pt', 
            map_location=torch.device(device)
        )
    )
    tokenizer = tokenizers.ABtokenizer()
    
    return AbLang, tokenizer, hparams
