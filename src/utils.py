# utils.py

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import io
from PIL import Image
import torch
import sklearn.metrics as metrics

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def smiles_to_image(smiles, size=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((size, size), dtype=np.uint8)
    img = Draw.MolToImage(mol, size=(size, size)).convert('L')  # grayscale
    return np.array(img, dtype=np.uint8) / 255.0

def build_smiles_vocab(smiles_list):
    charset = set()
    for smi in smiles_list:
        charset.update(list(smi))
    vocab = {ch: i+1 for i, ch in enumerate(sorted(charset))}  # +1 for padding at 0
    vocab['<pad>'] = 0
    return vocab

def tokenize_smiles(smiles, vocab, max_len):
    tokens = [vocab.get(ch, 0) for ch in smiles[:max_len]]
    if len(tokens) < max_len:
        tokens += [vocab['<pad>']] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

def compute_dti_metrics(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = np.sqrt(mse)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return {'rmse': rmse, 'pearson': corr}

def compute_se_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    auc = metrics.roc_auc_score(y_true, y_scores, average='micro')
    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    return {'auc_micro': auc, 'f1_micro': f1}
