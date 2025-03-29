
import os
import torch

def get_config():
    return {
        'data_path': 'Data',
        'embedding_model_name': 'all-mpnet-base-v2',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    }