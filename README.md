# Muon-Optimized Sparse MoE LLM
### A Reproducible Research Implementation Comparing Muon vs. AdamW Optimizers

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Research Paper](https://img.shields.io/badge/📄_Read_Research_Paper-PDF-red?style=for-the-badge)](https://github.com/MassYadav/Muon-MoE-LLM/blob/main/Research_pPaper/Manish_Yadav_Muon_vs_AdamW_Research.pdf)

## 🚀 Abstract
This repository contains a clean, modular implementation of a **Sparse Mixture-of-Experts (MoE) Transformer** built from first principles in PyTorch. The primary contribution of this work is a comparative analysis of training dynamics between the **Muon (Momentum Orthogonalized)** optimizer and the industry-standard **AdamW**, specifically in the context of sparse expert architectures.

Unlike standard implementations, this codebase features a custom **hybrid optimization schedule** and **Top-2 gating mechanisms** designed to mitigate expert collapse during pre-training.

**[>> Click here to read the full Research Paper (PDF)](https://github.com/MassYadav/Muon-MoE-LLM/blob/main/Research_pPaper/Manish_Yadav_Muon_vs_AdamW_Research.pdf)**

---

## 🔬 Methodology & Key Features

### 1. Sparse Mixture-of-Experts (MoE) Architecture
* **Dynamic Routing:** Implemented a **Top-2 Noisy Router** (`models/moe_llm.py`) that dynamically routes tokens to the best experts, scaling parameter count while keeping inference FLOPs constant.
* **Expert Collapse Mitigation:** Integrated **Auxiliary Load Balancing Loss** to penalize the router if it over-utilizes a single expert, ensuring uniform token distribution across expert networks.
* **Modern Components:** Utilizes **Rotary Positional Embeddings (RoPE)** and **Pre-RMSNorm** for enhanced training stability at depth.

### 2. The Muon Optimizer (Custom Implementation)
* **Orthogonal Updates:** Implemented the Muon update rule (`optimizers/muon.py`) which applies **Newton-Schulz iterations** to orthogonalize weight updates, enabling faster convergence on 2D matrices.
* **Hybrid Scheduling strategy:**
    * **Muon:** Applied to 2D internal projections (Attention & FFN weights).
    * **AdamW:** Applied to 1D embeddings and normalization layers (where Muon is historically unstable).

---

## 📊 Experimental Results
*(Detailed ablation studies and loss curves are available in the attached PDF)*

| Metric | AdamW (Baseline) | Muon (Optimized) | Result |
| :--- | :--- | :--- | :--- |
| **Convergence Speed** | Baseline | **~1.5x Faster** | Muon reached target loss in significantly fewer steps. |
| **Expert Utilization** | High Variance | **Balanced** | Auxiliary loss successfully distributed load. |
| **Memory Efficiency** | Standard | **High** | Hybrid optimizer reduced effective VRAM usage during training. |

> **Research Insight:** The Muon optimizer demonstrated superior wall-clock convergence on the MoE architecture, primarily due to its ability to normalize update steps across the sparse expert layers.

---

## 📂 Repository Structure

```bash
├── Research_pPaper/         # Full Technical Report
│   └── Manish_Yadav_Muon_vs_AdamW_Research.pdf
├── configs/                 # Dataclass-based configurations
│   ├── moe_config.py        # Model depth, Heads, Experts, Top-k settings
│   └── dataset_config.py    # Context Window & Tokenizer settings
├── data/                    # Streaming Data Pipeline
│   └── dataloader.py        # Infinite streaming dataloader
├── models/                  # MoE Architecture Source
│   ├── layers.py            # Transformer Blocks
│   └── moe_llm.py           # Final MoE Model Assembly
├── optimizers/              # Optimizer Algorithms
│   ├── muon.py              # Custom Muon Implementation
│   └── adam.py              # Reference AdamW Implementation
└── train_moe.py             # CLI Entry Point

## 🛠️ Usage
1. Installation

git clone [https://github.com/MassYadav/Muon-MoE-LLM.git](https://github.com/MassYadav/Muon-MoE-LLM.git)
cd Muon-MoE-LLM
pip install -r requirements.txt

2. Run Training Benchmark
To reproduce the Muon training results:

python train_moe.py --config configs/moe_config.py --optimizer muon --batch_size 32

To run the AdamW baseline:
python train_moe.py --config configs/moe_config.py --optimizer adam --batch_size 32

---

## 📜 Citation
If you use this code or findings, please cite the attached research report:

```bibtex
@article{yadav2025muonmoe,
  title={Analysis of Muon Optimization on Sparse Mixture-of-Experts Transformers},
  author={Manish Yadav},
  journal={GitHub Repository},
  year={2025},
  url={[https://github.com/MassYadav/Muon-MoE-LLM](https://github.com/MassYadav/Muon-MoE-LLM)}
}