# evaluate.py

import torch
from model import MultiTaskModel
from dataset import BindingDBDataset, SIDERDataset
from utils import compute_dti_metrics, compute_se_metrics
from torch.utils.data import DataLoader
import numpy as np

# Load test data
binding_test = []  # [{'smiles': ..., 'protein_id': ..., 'pKi': ...}]
sider_test = []    # [{'smiles': ..., 'side_effects': [...]}]
protein_embeddings = {}
smiles_vocab = {}
SE_CLASSES = 1000
MAX_LEN = 150
BATCH_SIZE = 32

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

binding_dataset = BindingDBDataset(binding_test, smiles_vocab, protein_embeddings, MAX_LEN)
sider_dataset = SIDERDataset(sider_test, smiles_vocab, SE_CLASSES, MAX_LEN)

binding_loader = DataLoader(binding_dataset, batch_size=BATCH_SIZE)
sider_loader = DataLoader(sider_dataset, batch_size=BATCH_SIZE)

model = MultiTaskModel(vocab_size=len(smiles_vocab), se_output_dim=SE_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("multi_task_model.pt", map_location=DEVICE))
model.eval()

# DTI Evaluation
y_true_dti, y_pred_dti = [], []
with torch.no_grad():
    for imgs, smiles, prots, pkis in binding_loader:
        imgs, smiles, prots = imgs.to(DEVICE), smiles.to(DEVICE), prots.to(DEVICE)
        preds = model.forward_dti(imgs, smiles, prots)
        y_true_dti.extend(pkis.numpy())
        y_pred_dti.extend(preds.cpu().numpy())

metrics_dti = compute_dti_metrics(np.array(y_true_dti), np.array(y_pred_dti))
print("DTI Test Metrics:", metrics_dti)

# SE Evaluation
y_true_se, y_score_se = [], []
with torch.no_grad():
    for imgs, smiles, labels in sider_loader:
        imgs, smiles = imgs.to(DEVICE), smiles.to(DEVICE)
        preds = model.forward_se(imgs, smiles)
        y_true_se.extend(labels.numpy())
        y_score_se.extend(torch.sigmoid(preds).cpu().numpy())

metrics_se = compute_se_metrics(np.array(y_true_se), np.array(y_score_se))
print("SE Test Metrics:", metrics_se)
