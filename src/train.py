# train.py
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import MultiTaskModel
from dataset import BindingDBDataset, SIDERDataset
from utils import compute_dti_metrics, compute_se_metrics

import random
import numpy as np

binding_df_path = "../data/bindingdb.csv"
binding_df = pd.read_csv(binding_df_path)
bindingdb_data = []
for idx, row in binding_df.iterrows():
    temp = {}
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']), canonical=True)
    protien_id = row["Target Name"]
    pki = row["Ki (nM)"]
    temp["smiles"] = smiles
    temp["protein_id"] = protien_id
    temp["pKi"] = pki
    bindingdb_data.append(temp)

protein_ids = binding_df["Target Name"].dropna().unique()
embedding_dim = 128
protein_embeddings = {pid: np.random.randn(embedding_dim) for pid in protein_ids}

smiles_list = binding_df['SMILES'].dropna().tolist()
charset = set()
for smi in smiles_list:
    charset.update(list(smi))
smiles_vocab = {char: idx for idx, char in enumerate(sorted(charset), start=1)}



sider_data = []      # List of dicts: {'smiles': ..., 'side_effects': [...]} 

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
LAMBDA = 1.0
MAX_LEN = 150
SE_CLASSES = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create datasets and loaders
binding_dataset = BindingDBDataset(bindingdb_data, smiles_vocab, protein_embeddings, MAX_LEN)
sider_dataset = SIDERDataset(sider_data, smiles_vocab, SE_CLASSES, MAX_LEN)

binding_loader = DataLoader(binding_dataset, batch_size=BATCH_SIZE, shuffle=True)
sider_loader = DataLoader(sider_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MultiTaskModel(vocab_size=len(smiles_vocab), se_output_dim=SE_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    iter_dti = iter(binding_loader)
    iter_se = iter(sider_loader)

    dti_loss_total, se_loss_total, steps = 0.0, 0.0, 0

    while True:
        try:
            dti_batch = next(iter_dti)
        except StopIteration:
            iter_dti = iter(binding_loader)
            dti_batch = next(iter_dti)

        try:
            se_batch = next(iter_se)
        except StopIteration:
            iter_se = iter(sider_loader)
            se_batch = next(iter_se)

        # DTI batch
        imgs, smiles, prots, pkis = [x.to(DEVICE) for x in dti_batch]
        preds_dti = model.forward_dti(imgs, smiles, prots)
        loss_dti = mse_loss(preds_dti, pkis)

        # SE batch
        imgs2, smiles2, labels = [x.to(DEVICE) for x in se_batch]
        preds_se = model.forward_se(imgs2, smiles2)
        loss_se = bce_loss(preds_se, labels)

        loss = loss_dti + LAMBDA * loss_se
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dti_loss_total += loss_dti.item()
        se_loss_total += loss_se.item()
        steps += 1

    print(f"Epoch {epoch+1} | DTI Loss: {dti_loss_total/steps:.4f} | SE Loss: {se_loss_total/steps:.4f}")

# Save model after training
torch.save(model.state_dict(), "multi_task_model.pt")
