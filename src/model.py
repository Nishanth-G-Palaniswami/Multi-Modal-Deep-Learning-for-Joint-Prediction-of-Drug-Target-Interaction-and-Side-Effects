# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DrugImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.cnn(x)  # [B, 128, 1, 1]
        return x.view(x.size(0), -1)  # [B, 128]

class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, emb_dim))

    def forward(self, x):  # x: [B, L]
        B, L = x.shape
        x = self.emb(x) + self.pos_emb[:, :L]
        x = x.transpose(0, 1)  # [L, B, E]
        x = self.transformer(x)
        return x.mean(dim=0)  # [B, E]

class MultiTaskModel(nn.Module):
    def __init__(self, vocab_size, se_output_dim, protein_dim=1024):
        super().__init__()
        self.image_encoder = DrugImageEncoder()
        self.smiles_encoder = SmilesEncoder(vocab_size)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.dti_head = nn.Sequential(
            nn.Linear(256 + protein_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
        self.se_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, se_output_dim)
        )

    def forward_dti(self, img, smiles, prot):
        img_feat = self.image_encoder(img)
        smiles_feat = self.smiles_encoder(smiles)
        drug_feat = self.fusion(torch.cat([img_feat, smiles_feat], dim=1))
        combined = torch.cat([drug_feat, prot], dim=1)
        return self.dti_head(combined).squeeze(1)

    def forward_se(self, img, smiles):
        img_feat = self.image_encoder(img)
        smiles_feat = self.smiles_encoder(smiles)
        drug_feat = self.fusion(torch.cat([img_feat, smiles_feat], dim=1))
        return self.se_head(drug_feat)
