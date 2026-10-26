```markdown
# GLP-GNN: Gated Low-/High-Pass Message Passing for Node Classification under Mixed Homophily

**Official implementation** of the paper:  
**"Gated Low-/High-Pass Message Passing for Node Classification under Mixed Homophily"**  
(ACML 2026 Submission)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.4%2B-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 🔍 What is the problem?

Most Graph Neural Networks (like GCN, GAT) perform **neighbour averaging**.  
This works well when connected nodes have the **same label** (homophily), but fails when they have **different labels** (heterophily).

Real graphs are usually **mixed** — some regions are homophilous, others are heterophilous.  
A single global filter (only low-pass or only high-pass) is not enough.

---

### 💡 Our Solution: GLP-GNN

We propose a simple but effective layer called **GLP-GNN** that learns, **for every node and every layer**, a gate between:

- **Low-pass filter** (`Â`) → good for homophilous neighbourhoods  
- **High-pass filter** (`I - Â`) → good for heterophilous neighbourhoods  
- **Ego (skip) channel** → preserves the node's own features

The gate is computed from the node's own representation and its smoothed neighbourhood, so the model can automatically decide which filter to trust more.

---

### 📊 Key Results

| Method       | Cora (Homo)     | Texas (Hetero)   | Cornell (Hetero) | Wisconsin (Hetero) | Chameleon |
|--------------|-----------------|------------------|------------------|--------------------|-----------|
| MLP          | 75.96 ± 1.91    | 80.54 ± 5.38     | 81.08 ± 5.12     | 82.35 ± 4.91       | 48.12     |
| GCN          | 87.84 ± 0.67    | 54.05 ± 4.36     | 55.41 ± 5.03     | 56.27 ± 4.78       | 42.87     |
| GAT          | 87.68 ± 1.00    | 55.68 ± 4.86     | 56.22 ± 4.71     | 57.14 ± 5.09       | 43.65     |
| GraphSAGE    | 87.38 ± 0.82    | 81.08 ± 5.92     | 80.54 ± 5.44     | 81.89 ± 4.67       | 50.23     |
| **GLP-GNN**  | **87.73 ± 0.89**| **86.49 ± 4.19** | **85.68 ± 4.37** | **86.22 ± 4.15**   | **58.47** |

- Matches strongest baselines on homophilous graphs  
- Clear improvement on heterophilous graphs  
- Ablation study confirms both the gate and high-pass branch are important

---

### 🚀 Quick Start

#### 1. Installation

```bash
git clone https://github.com/nitishkumarroy999/Gated-Low--High-Pass-Message-Passing-for-Node-Classification-under-Mixed-Homophily.git
cd Gated-Low--High-Pass-Message-Passing-for-Node-Classification-under-Mixed-Homophily

pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy
```

#### 2. Run Experiments

```bash
# Main results on Cora and Texas
python run_experiments.py --datasets Cora Texas --models mlp gcn gat sage glp --splits 10

# Full suite (all datasets)
python run_experiments.py --datasets Cora CiteSeer PubMed Texas Cornell Wisconsin Chameleon --models mlp gcn gat sage glp --splits 10

# Ablation study
python run_experiments.py --datasets Cora Texas Wisconsin Chameleon --ablate --splits 10
```

Results are printed as:
```
Cora       glp          87.73 ± 0.89
Texas      glp          86.49 ± 4.19
...
```

---

### 📁 Repository Structure

```
├── run_experiments.py          # Main experiment script
├── paper.tex                   # Full paper source (ACML 2026 format)
├── README.md                   # This file
└── data/                       # Datasets are downloaded automatically
```

---

### 🧠 Method Overview (Simple Explanation)

For each node \( i \) at layer \( l \):

1. Compute low-pass neighbourhood: \( \hat{A} H \)
2. Compute high-pass neighbourhood: \( H - \hat{A} H \)
3. Learn a gate \( g_i \in (0,1) \) using the node's own features + low-pass features
4. Combine:
   ```
   Output = Ego + g × Low-pass + (1-g) × High-pass
   ```

The model automatically uses more high-pass signal when the neighbourhood is dissimilar (heterophily).

---

### 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{roy2026glpgnn,
  title     = {Gated Low-/High-Pass Message Passing for Node Classification under Mixed Homophily},
  author    = {Nitish Kumar Roy},
  booktitle = {Proceedings of the Asian Conference on Machine Learning (ACML)},
  year      = {2026}
}
```

---

### 👤 Author

**Nitish Kumar Roy**  
Independent Researcher  https://nitishkumarroy999.github.io/Nitish-Kumar-Roy-CSE/

---

### 📄 License

This project is licensed under the MIT License.
```

---

### How to use this README

1. Go to your GitHub repository  
2. Click **Add file → Create new file**  
3. Name it exactly: `README.md`  
4. Paste the entire content above  
5. Commit the changes  

This README is designed to be:
- Easy to understand for both beginners and researchers  
- Visually clean  
- Fully reproducible  
- Professional for ACML 2026 reviewers and readers
