import numpy as np
from tqdm import tqdm
import pandas as pd
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder, ProtTransT5XLU50Embedder


data_path = 'test.tsv'
embeddings = dict()
embedder  = ProtTransT5XLU50Embedder(model_directory = './embedding_model/prottrans_t5_xl_u50')
with open(data_path, 'r') as f:
    text = f.read()
lines = text.split('\n')

for l in tqdm(lines):
    p = l.split(' ')
    print(p[1], type(p[1]))
    embedding = embedder.emded(p[1])
    embeddings[p[0]] = np.sum(embedding, axis=0)
np.savez('d_prot_embed.npz', **embeddings)