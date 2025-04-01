
import os
import torch

def get_config():
    return {
        'data_path': 'Data',
        'embedding_model_name': 'all-mpnet-base-v2',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'generator_model_name': 'google/gemma-2b-it',
        'use_quantization_config': False,
        'attn_implementation': 'sdpa',
        'max_new_tokens': 512,
        'top_k': 5,

    }

