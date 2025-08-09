# dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import io

from utils import canonicalize_smiles, smiles_to_image, tokenize_smiles

class BindingDBDataset(Dataset):
    def __init__(self, data, smiles_vocab, protein_embeddings, max_seq_len=150):
        """
        data: List of dicts with keys: 'smiles', 'protein_id', 'pKi'
        smiles_vocab: dictionary mapping characters to indices
        protein_embeddings: dict mapping protein_id -> precomputed embedding
        """
        self.data = data
        self.vocab = smiles_vocab
        self.protein_embeddings = protein_embeddings
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        smiles = canonicalize_smiles(sample['smiles'])
        image = smiles_to_image(smiles)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, 100, 100]
        smiles_tensor = tokenize_smiles(smiles, self.vocab, self.max_seq_len)
        protein_embedding = self.protein_embeddings[sample['protein_id']]
        protein_tensor = torch.tensor(protein_embedding, dtype=torch.float32)
        pki = torch.tensor(sample['pKi'], dtype=torch.float32)
        return image_tensor, smiles_tensor, protein_tensor, pki


class SIDERDataset(Dataset):
    def __init__(self, data, smiles_vocab, num_side_effects, max_seq_len=150):
        """
        data: List of dicts with keys: 'smiles', 'side_effects' (list of indices)
        """
        self.data = data
        self.vocab = smiles_vocab
        self.num_side_effects = num_side_effects
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        smiles = canonicalize_smiles(sample['smiles'])
        image = smiles_to_image(smiles)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        smiles_tensor = tokenize_smiles(smiles, self.vocab, self.max_seq_len)
        label_vector = torch.zeros(self.num_side_effects, dtype=torch.float32)
        label_vector[sample['side_effects']] = 1.0
        return image_tensor, smiles_tensor, label_vector
