# Multi‑Modal Deep Learning for Joint Prediction of Drug–Target Interaction & Side‑Effects

> **One‑pager:** Multi‑modal, multi‑task model that jointly predicts **binding affinity (pKI)** and **side‑effects** from SMILES (sequence + 2D image) and **protein embeddings**.

<p align="center">
  <img src="assets/drug target interaction.png" alt="Drug–Target Interaction architecture" width="85%"/>
</p>

<p align="center">
  <img src="assets/side effect prediction.png" alt="Side‑Effect Prediction architecture" width="85%"/>
</p>

---

## ✨ Highlights

* **Multi‑modal inputs**: SMILES **sequence** → Transformer, SMILES **image** → CNN; **protein** sequences → ProtBERT + linear head.
* **Multi‑task learning**:

  * **DTI**: regression on **pKI**
  * **Side‑effects**: multi‑label classification
* **Fusion**: image + text → drug embedding → fused with protein embedding.
* **Datasets**: **BinderDB** (affinity) and **SIDER** (side‑effects), protein sequences via **UniProt**.

## 📦 Repository Structure

```
src/
 ├─ dataset.py        # Data loading & preprocessing (SMILES, images, proteins)
 ├─ model.py          # CNN, Transformer, ProtBERT head, fusion, multitask heads
 ├─ train.py          # Training loop, multitask losses (MSE + BCE), metrics
 ├─ evaluate.py       # Eval for DTI (RMSE/MSE) & SE (mAP/F1/partial acc)
 ├─ utils.py          # Helpers (seeding, logging, checkpointing)
 ├─ sdf_reader.py     # SDF/SMILES utilities (RDKit)
 └─ config.json       # Hyperparams & paths
```

## 🔧 Setup

```bash
# Python 3.9+
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

**Core deps:** `torch`, `transformers`, `rdkit-pypi`, `pandas`, `numpy`, `scikit-learn`, `opencv-python`

Create a `.env` or edit `config.json` to point to your data folders.

## 🗂️ Data

* **BinderDB** → drug–target affinity (Ki → **pKI = −log10(Ki)**)
* **SIDER** → side‑effects for \~1.4k drugs mapped to 5,880 terms
* **UniProt/ProtBERT** → protein sequence embeddings (frozen or fine‑tuned)

Convert SMILES to 2D images with **RDKit**; tokenized SMILES feed the Transformer.

## ▶️ Train

```bash
python src/train.py --config src/config.json
```

Key config knobs: batch size, learning rate, loss weights, image size (e.g., 64×64), transformer dims, ProtBERT freeze.

## 📈 Evaluate

```bash
python src/evaluate.py --checkpoint runs/best.ckpt --split test
```

**DTI**: RMSE/MSE on pKI.
**SE**: mAP/F1/partial‑accuracy.

## 🧪 Results (course project)

* **SE head** benefited from multitask learning; **DTI** regressor mildly regressed due to gradient conflict.
* Next: apply **PCGrad/GradNorm** for conflict mitigation; explore **pKd/IC50**.

## 🗺️ Roadmap

* [ ] Gradient surgery (PCGrad) for multi‑task stability
* [ ] Fine‑tune ProtBERT end‑to‑end
* [ ] Add pKd/IC50 objectives & uncertainty calibration
* [ ] Weights & Biases logging + model cards

## 📝 Citation

> Yu et al., **Gradient Surgery for Multi‑Task Learning**, arXiv:2001.06782.

## 👥 Authors

* **Nishanth G. Palaniswami (NG3124)**
* **Arun Purohit (AP9111)**

## 📄 License

MIT
