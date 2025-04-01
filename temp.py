import os
from spacy.lang.en import English # see https://spacy.io/usage for install instructions
from sentence_transformers import SentenceTransformer
import torch

example = "This is an example sentence. This is another sentence. This is a third sentence."
model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

embedding = model.encode(example, convert_to_tensor=True).unsqueeze(0)
print(embedding.shape)